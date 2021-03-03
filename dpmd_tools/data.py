"""Helper module with dpdata subclasses."""

from concurrent.futures import as_completed
from copy import deepcopy
from pathlib import Path
from warnings import warn
from typing import (
    Optional,
    TYPE_CHECKING,
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Union,
    ValuesView,
    Generic,
    TypeVar,
)
import os

import numpy as np
from dpdata import LabeledSystem, MultiSystems
from tqdm import tqdm
from typing_extensions import Literal, TypedDict
from loky import get_reusable_executor
from dpmd_tools.utils import split_into

if TYPE_CHECKING:

    _DATA = TypedDict(
        "_DATA",
        {
            "atom_names": np.ndarray,
            "atom_numbs": np.ndarray,
            "atom_types": np.ndarray,
            "cells": np.ndarray,
            "coords": np.ndarray,
            "energies": np.ndarray,
            "forces": np.ndarray,
            "virials": np.ndarray,
            # the following are custom
            "used": np.ndarray,
            "clusters": Optional[np.ndarray],
            "iteration": Optional[np.ndarray],
        },
    )

MAX_FRAMES_PER_SET = 5000
_SYS_TYPE = TypeVar("_SYS_TYPE", "LabeledSystemMask", "LabeledSystemSelected")


class AllSelectedError(Exception):
    """Raised when all structures from current dataset have already been selected."""

    pass


class MultiSystemsVar(MultiSystems, Generic[_SYS_TYPE]):
    """Conveniens subclass of Multisystems.

    Adds `pathlib.Path` compatibility and few other convenience methods.
    Also introduces better auto set partitioning.
    """

    def to_deepmd_npy(
        self,
        folder: Path,
        set_size: Union[int, list, Literal["auto"]] = 5000,
        prec=np.float32,
    ):
        """Dump the system in deepmd compressed format.

        Dump numpy binary (.npy) to `folder` for each system.

        Parameters
        ----------
        folder : str
            The output folder
        set_size : int
            The size of each set, if `auto` than the system will be partitioned
            to as equally long sets as possible while also trying to maintain
            data split as close to 90:10 and strictly obeying condition for
            MAX_FRAMES_PER_SET
        prec: {numpy.float32, numpy.float64}
            The floating point precision of the compressed data
        """
        if isinstance(set_size, int):
            set_size = [set_size for _ in range(self.__len__())]
        elif set_size == "auto":
            set_size = partition_systems(self)
        else:
            assert len(set_size) == self.__len__(), "Incorrect length of set_size"

        dump_job = tqdm(
            zip(set_size, self.systems.items()), ncols=100, total=len(set_size)
        )

        for ss, (system_name, system) in dump_job:
            system.to_deepmd_npy(
                str((folder / system_name).resolve()), set_size=ss, prec=prec
            )

    def to_deepmd_raw(self, folder: Path):
        for system_name, system in self.systems.items():
            path = folder / system_name
            system.to_deepmd_raw(path)

    def keys(self) -> KeysView[str]:
        return self.systems.keys()

    def items(self) -> ItemsView[str, _SYS_TYPE]:
        return self.systems.items()

    def values(self) -> ValuesView[_SYS_TYPE]:
        return self.systems.values()

    def shuffle(self):
        for system in self.values():
            system.shuffle()

    @property
    def iteration(self) -> int:
        it = set([s.iteration for s in self.values()])
        if len(it) > 1:
            warn(
                f"not all subsystems have the same number of selection iterations: {it}"
                f"this concerns only graph export names, so mostly you can safely "
                f"ignore this message"
            )

        return max(it)

    def predict(self, graphs: List[Path]) -> Iterator["MultiSystemsVar"]:

        for g in graphs:
            prediction = super().predict(str(g.resolve()))
            multi_sys = MultiSystemsVar()

            # copy data
            for attr in vars(prediction):
                setattr(multi_sys, attr, getattr(prediction, attr))

            yield multi_sys

    def collect_single(
        self, paths: List[Path], dir_process: Callable[[Path], List[_SYS_TYPE]]
    ):
        """Single core serial data collector."""
        futures = tqdm(paths, ncols=100, total=len(paths))

        for p in futures:

            try:
                systems = dir_process(p)
            except Exception as e:
                futures.write(f"Error in {p.name}: {e}")
            else:
                for s in systems:
                    try:
                        self.append(s)
                    except Exception as e:
                        futures.write(f"Error in {p.name}: {e}")

    def collect_cf(
        self, paths: List[Path], dir_process: Callable[[Path], List[_SYS_TYPE]],
    ):
        """Parallel async data collector."""
        print("All atom types will be changed to Ge!!!")

        with get_reusable_executor(max_workers=10) as pool:
            future2data = {pool.submit(dir_process, p): p for p in paths}

            futures = tqdm(as_completed(future2data), ncols=100, total=len(paths))
            for future in futures:
                path = future2data[future].name
                futures.set_description(f"extracting: {path}")
                try:
                    systems = future.result()
                except AllSelectedError as e:
                    futures.write(
                        f"All structures from system {path} have already been selected,"
                        f" skipping..."
                    )
                except Exception as e:
                    futures.write(f"Error in {path}: {e}")
                else:
                    for s in systems:
                        try:
                            s.data["atom_names"] = ["Ge"]
                            self.append(s)
                        except Exception as e:
                            futures.write(f"Error in {path}: {e}")


class LabeledSystemMask(LabeledSystem):
    """Masked subclass that remembers which frames were already used.

    Together with raw files it will export `used.raw` which will contain a
    column for each iteration marking which structures were used:

    it1 it2 it3 mask
    === === === ====
    0   1   0   1
    1   0   0   1
    0   0   1   1
    0   1   0   1
    0   0   0   0
    1   0   0   1

    Zeros mark unused structures and ones that where already used. These cannot
    be used again.
    `used.raw` file is exported only when exporting to raw.
    Typical workflow is to load some data format and use a portion of files for
    training. Everything is exported to raw and then in the next iteration one
    reads agin all the files exported in raw and selects some new frames to add
    to dataset whic where not used before

    Note
    ----
    `used` data key must be always present if file is not found zero filled array will
    be created, meaning taht none of the structures was used before.
    `clusters` data key is not mandatory, if the clusters file is present in directory
    it is read. If it is not present cluster selection criteria will fail.
    """

    data: "_DATA"

    def __init__(  # NOSONAR
        self,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional["_DATA"] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            file_name=str(file_name) if file_name else None,
            fmt=fmt,
            type_map=type_map,
            begin=begin,
            step=step,
            data=data,
            **kwargs,
        )

        # check format
        # if reading from raw assume used.raw is already in-place
        if "raw" in fmt:

            if not file_name:
                raise ValueError("must pass in filename")
            used_file = file_name / "used.raw"

            #  check if file exists
            if not used_file.is_file():
                raise FileNotFoundError(f"there is no used.raw file in: {used_file}")

            # load
            self.data["used"] = np.loadtxt(used_file)

            # check if it i column vector(s)
            if len(self.data["used"].shape) == 1:
                self.data["used"] = np.atleast_2d(self.data["used"]).T

            # checkt if dimensions are right
            if not self.data["used"].shape[0] == self.data["cells"].shape[0]:
                raise ValueError(
                    f"used.raw file has wrong dimensions, expected :"
                    f"{self.data['cells'].shape[0]} rows, got "
                    f"{self.data['used'].shape[0]}"
                )
        # else init from scratch but not if data with 'used' key are passed in
        else:
            try:
                data["used"]  # type: ignore
            except (KeyError, TypeError):
                self.data["used"] = np.zeros(
                    (self.data["cells"].shape[0], 1), dtype=int
                )

        if file_name is not None:
            if (file_name / "clusters.raw").is_file():
                self.data["clusters"] = np.loadtxt(file_name / "clusters.raw")
            else:
                self.data["clusters"] = None

        # cond > 1 is for predict when we generate subsystems of length == 1
        if (
            np.count_nonzero(self.mask) >= len(self.data["cells"])
            and len(self.mask) > 1
        ):
            raise AllSelectedError("All structures are alread selected")

    def to_deepmd_raw(self, folder: Path):

        files_exist = [
            (folder / f"{name}.raw").is_file()
            for name in (
                "box",
                "coord",
                "energy",
                "force",
                "type",
                "type_map",
                "virial",
            )
        ]
        if not all(files_exist):
            super().to_deepmd_raw(folder)

        if self.data["used"] is not None:
            np.savetxt(folder / "used.raw", self.data["used"], fmt="%d")

    @property
    def iteration(self) -> int:
        """Return curent selection iteration."""
        return self.data["used"].shape[1]

    def has_clusters(self) -> bool:
        return "clusters" in self.data

    @property
    def clusters(self) -> Optional[np.ndarray]:
        """Get cluster index for all structures."""
        return self.data["clusters"]

    def append_mask(self, selected_indices: np.ndarray):
        """Add mask resulting from new selection iteration."""
        # create used structures mask for cuttent iteration
        mask = np.zeros(self.get_nframes(), dtype=int)
        mask[selected_indices] = 1

        if self.data["used"].shape[1] == 1 and self.data["used"].sum() == 0:
            self.data["used"] = np.atleast_2d(mask).T
        else:
            self.data["used"] = np.hstack((self.data["used"], np.atleast_2d(mask).T))

    def append(self, system: "LabeledSystemMask"):
        super().append(system)
        self.data["used"] = np.concatenate((self.data["used"], system["used"]), axis=0)
        if self.has_clusters and system.has_clusters is not None:
            self.data["clusters"] = np.concatenate(
                (self.data["clusters"], system.data["clusters"]), axis=0
            )

    def get_subsystem_indices(self, iteration: Optional[int] = None) -> np.ndarray:
        """Get indices of frames filtered by iterations.

        Parameters
        ----------
        iteration: Optional[int]
            filter by iteration of this number and all the previous ones

        Returns
        -------
        np.ndarray
            indices filtered by passed in iteration
        """
        return np.argwhere(self._mask(iteration) == 1).flatten()

    @property
    def mask(self) -> np.ndarray:
        """Generate mask from used array.

        All rows which contain at least one '1' have 1 in mask.
        Only frame that was not used in any iteration has 0.
        See class docstring for example.
        """
        return self._mask(None)

    def _mask(self, iteration: Optional[int]) -> np.ndarray:
        """Outputs mask which has ones for selected structs in specified iterations."""
        if iteration is None:  # result from all selection iterations
            return np.array(~(self.data["used"][:, :None] == 0).all(axis=1), dtype=int)
        elif iteration == 0:  #  ignore selections in previous iterations
            return np.zeros(self.data["used"].shape[0])
        elif iteration > 0:  # select accordingly to iteration number
            return np.array(
                ~(self.data["used"][:, :iteration] == 0).all(axis=1), dtype=int
            )
        else:
            n_cols = self.data["used"].shape[1]
            raise ValueError(
                f"iteration number must be in range 0, {n_cols - 1} or None"
            )

    def sub_system(self, f_idx: Union[np.ndarray, int]) -> "LabeledSystemMask":
        tmp_sys = super().sub_system(f_idx)
        tmp_sys.data["used"] = self.data["used"][f_idx]
        if isinstance(f_idx, int):
            tmp_sys.data["used"] = np.atleast_2d(tmp_sys.data["used"])

        return LabeledSystemMask(data=tmp_sys.data)

    def get_subsystem(self, iteration: Optional[int] = None) -> "LabeledSystemSelected":
        """Construct a subsystem from the system.

        Use mask and iteration number to filter frames.

        For subsystem compute and store generation in which was each frame added. The
        number of indices in each generations is aquired simply from
        `get_subsystem_indices(iteration)` method. Than an array with this lenght and
        all elements == generation number is generated. These arrays are then
        concatenated and sorted accorging to the subsystem indices array. See examples
        from detailed expalanation.

        Parameters
        ----------
        iteration: Optional[int]
            filter by iteration of this number and all the previous ones

        Returns
        -------
        sub_system : LabeledSystemMask
            The masked subsystem

        Examples
        --------
        Generation indices:

        >>> subsys_idx = [1, 5, 8, 12, 17]
        >>> subsys_idx_it1 = [1, 8, 12]
        >>> subsys_idx_it2 = [5, 17]
        >>> iterations = [[1, 1, 1], [2, 2]] => [1, 1, 1, 2, 2]
        >>> iteration_right_order = [1, 2, 1, 1, 2]  # sorted by subsys_idx
        """
        f_idx = self.get_subsystem_indices(iteration)
        tmp_sys = LabeledSystemSelected(data=super().sub_system(f_idx).data)

        if iteration is None:
            iteration = self.data["used"].shape[1]

        iteration_idx = []
        frame_idx = []  # these are the same a f_idx but ordered by iteration
        for i in range(iteration):
            # get iteration index from mask, simply generate array filled with
            # iteration index with length corresponding to number of frames selected in
            # that iteration. See docstrings
            shape = np.count_nonzero(self.data["used"][:, i])
            idx = np.full(shape, i + 1, dtype=int)
            iteration_idx.append(idx)

            # now get position indices of frames used in that iteration
            idx = np.argwhere(self.data["used"][:, i]).flatten()
            frame_idx.append(idx)

        iteration_indices = np.concatenate(iteration_idx)
        frame_indices = np.concatenate(frame_idx)
        tmp_sys.data["iteration"] = iteration_indices[frame_indices.argsort()]
        return tmp_sys

    def predict(self, dp: Path):
        """Predict energies and forces by deepmd-kit for yet unselected structures.

        Parameters
        ----------
        dp : deepmd.DeepPot or str
            The deepmd-kit potential class or the filename of the model.

        Returns
        -------
        labeled_sys LabeledSystemMask
            The labeled system.
        """
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        import deepmd.DeepPot as DeepPot

        deeppot = DeepPot(str(dp))
        type_map = deeppot.get_type_map()
        ori_sys = self.copy()
        ori_sys.sort_atom_names(type_map=type_map)
        atype = ori_sys["atom_types"]

        labeled_sys = self.copy()

        recompute_idx = self.get_subsystem_indices()

        for idx in tqdm(
            range(self.get_nframes()), total=self.get_nframes(), ncols=100, leave=False
        ):

            ss = super().sub_system(idx)

            coord = ss["coords"].reshape((-1, 1))
            if not ss.nopbc:
                cell = ss["cells"].reshape((-1, 1))
            else:
                cell = None

            data = ss.data

            if idx in recompute_idx:
                # already selected in previous iteration -> skip
                # create fake data, it does not matter since these will be skipped
                # anyway because of the mask
                data["energies"] = np.zeros((1, 1))
                data["forces"] = np.zeros((1, ss.get_natoms(), 3))
                data["virials"] = np.zeros((1, 3, 3))
            else:
                e, f, v = deeppot.eval(coord, cell, atype)
                data["energies"] = e.reshape((1, 1))
                data["forces"] = f.reshape((1, -1, 3))
                data["virials"] = v.reshape((1, 3, 3))

            labeled_sys.data["energies"][idx] = data["energies"]
            labeled_sys.data["forces"][idx] = data["forces"]
            labeled_sys.data["virials"][idx] = data["virials"]
        return labeled_sys


class LabeledSystemSelected(LabeledSystemMask):
    """System of selected structures for training from all selection iterations.

    Has additional data field `iteration` which tells in waht iteration was the
    respective structure selected.
    """

    def __init__(
        self,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional["_DATA"] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            file_name=file_name if file_name else None,
            fmt=fmt,
            type_map=type_map,
            begin=begin,
            step=step,
            data=data,
            **kwargs,
        )
        self.data["iteration"] = None

    def append(self, system: "LabeledSystemSelected"):
        super().append(system)
        if self.data["iteration"] is not None and system["iteration"] is not None:
            self.data["iteration"] = np.concatenate(
                (self.data["iteration"], system["iteration"]), axis=0
            )

    def shuffle(self):
        """Also shuffle labeled data e.g. energies and forces."""
        idx = super().shuffle()
        self.data["iteration"] = self.data["iteration"][idx]
        return idx

    @property
    def iteration(self):
        return self.data["iteration"].max()

    def to_deepmd_raw(self, folder: Path):

        self.data["used"] = None
        super().to_deepmd_raw(folder)

        if self.data["iteration"] is not None:
            np.savetxt(folder / "iteration.raw", self.data["iteration"], fmt="%d")

    def copy(self):
        tmp_sys = super().copy()
        tmp_sys.data["iteration"] = deepcopy(self.data["iteration"])
        return tmp_sys

    def sub_system(self, f_idx: Union[np.ndarray, int]) -> "LabeledSystemMask":
        tmp_sys = super().sub_system(f_idx)
        tmp_sys.data["iteration"] = self.data["iteration"][f_idx]
        if isinstance(f_idx, int):
            tmp_sys.data["iteration"] = np.atleast_2d(tmp_sys.data["iteration"])

        return LabeledSystemSelected(data=tmp_sys.data)


def partition_systems(multi_sys: MultiSystemsVar) -> List[int]:
    """Inteligently partition systems to sets.

    Instead of 10, 10, 7 that is output by dpdata -> 9, 9, 9
    Try to preserve 10% test data ratio with max set size == 5000.

    Parameters
    ----------
    multi_sys : MultiSystemsVar
        dpdata mustisystem class

    Returns
    -------
    List[int]
        list with partitioning for each system

    Raises
    ------
    RuntimeError
        If partitioning fails (too many structures)
    """
    d: LabeledSystem
    n_frames = []
    n_tests = []
    for name, d in multi_sys.items():
        d.shuffle()
        sys_frames = d.get_nframes()

        # ideally split to 10% portions the last will be used for testing
        # if 10% produces too big set size, lower that portion until the
        # condition is satified
        for parts in range(10, 100):
            sets = split_into(sys_frames, parts)

            set_frames = sets.pop(0)
            set_tests = sets.pop(len(sets) - 1)

            # the difference when e.g. split will be [30, 30, 29, 29]
            # dpdata will split to: [30, 30, 30, 28]
            # diff accounts for that
            diff = 0
            for i in range(len(sets)):
                diff += sets[0] - sets[i]

            set_tests -= diff

            if set_frames < MAX_FRAMES_PER_SET:
                print(
                    f"system {name} with {sys_frames} frames, will "
                    f"be partitioned to sets counting {set_frames}"
                )
                n_frames.append(set_frames)
                n_tests.append(set_tests)
                break
        else:
            raise RuntimeError(
                f"Couldn't split system {name} with {sys_frames} to parts "
                f"while satisfying the condition to keep set size under: "
                f"{MAX_FRAMES_PER_SET}"
            )

    return n_frames
