"""Helper module with dpdata subclasses."""

from concurrent.futures import as_completed
import math
from pathlib import Path
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
)

import numpy as np
from dpdata import LabeledSystem, MultiSystems
from tqdm import tqdm
from typing_extensions import Literal, TypedDict
from loky import get_reusable_executor

if TYPE_CHECKING:
    from ase import Atoms

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
            "used": np.ndarray,
            "clusters": Optional[np.ndarray],
        },
    )

MAX_FRAMES_PER_SET = 5000


class MultiSystemsVar(MultiSystems):
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
            assert len(set_size) == self.__len__(), "Incorrect length of " "set_size"

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

    def items(self) -> ItemsView[str, LabeledSystem]:
        return self.systems.items()

    def values(self) -> ValuesView[LabeledSystem]:
        return self.systems.values()

    def shuffle(self):
        for system in self.values():
            system.shuffle()

    def predict(self, graphs: List[Path]) -> Iterator["MultiSystemsVar"]:

        for g in graphs:
            prediction = super().predict(str(g.resolve()))
            multi_sys = MultiSystemsVar()

            # copy data
            for attr in vars(prediction):
                setattr(multi_sys, attr, getattr(prediction, attr))

            yield multi_sys

    def collect_single(
        self, paths: List[Path], dir_process: Callable[[Path], List[LabeledSystem]]
    ):
        """Single core serial data collector"""

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
        self,
        paths: List[Path],
        dir_process: Callable[[Path], List["LabeledSystemMask"]],
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
                except Exception as e:
                    futures.write(f"Error in {path}: {e}")
                else:
                    for s in systems:
                        try:
                            s.data["atom_names"] = ["Ge"]
                            self.append(s)
                        except Exception as e:
                            futures.write(f"Error innnn {path}: {e}")


class LabeledSystemMask(LabeledSystem):
    """This is masked subclass that remembers which frames were already used.

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
    """

    data: "_DATA"

    def __init__(
        self,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional[str] = None,
        used_file: Optional[Path] = None,
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

            # guess file if not passed in
            if not used_file:
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
                data["used"]
            except (KeyError, TypeError):
                self.data["used"] = np.zeros(
                    (self.data["cells"].shape[0], 1), dtype=int
                )

        if file_name is not None:
            if (file_name / "clusters.raw").is_file():
                self.data["clusters"] = np.loadtxt(file_name / "clusters.raw")
            else:
                self.data["clusters"] = None

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

        np.savetxt(folder / "used.raw", self.data["used"], fmt="%d")

    def append_mask(self, mask: np.ndarray):
        """Add mask resulting from new selection iteration."""
        if self.data["used"].shape[1] == 1 and self.data["used"].sum() == 0:
            self.data["used"] = np.atleast_2d(mask).T
        else:
            self.data["used"] = np.hstack((self.data["used"], np.atleast_2d(mask).T))

    def append(self, system: "LabeledSystemMask"):
        super().append(system)
        self.data["used"] = np.concatenate((self.data["used"], system["used"]), axis=0)
        if self.data["clusters"] is not None and system["clusters"] is not None:
            self.data["clusters"] = np.concatenate(
                (self.data["clusters"], system["clusters"]), axis=0
            )

    def get_subsystem_indices(self) -> np.ndarray:
        """Get indices of frames filtered by all iterations."""
        return np.argwhere(self.mask == 1).flatten()

    @property
    def mask(self) -> np.ndarray:
        """Generate mask from used array.

        All rows which contain at least one '1' have 1 in mask.
        Only frame that was not used in any iteration has 0.
        See class docstring for example.
        """
        return np.array(~(self.data["used"] == 0).all(axis=1), dtype=int)

    @property
    def clusters(self) -> np.ndarray:
        """Get cluster index for all structures."""

        return self.data["clusters"]

    def get_nframes_notused(self):
        """Get frames that where not already used in training.

        References
        ----------
        https://stackoverflow.com/questions/23726026/finding-which-rows-have-all-elements-as-zeros-in-a-matrix-with-numpy
        """
        return np.where((self.data["used"] == 0).all(axis=1))[0].shape[0]


def split_into(n: int, p: int):
    """Split number n to p parts as evenly as possible.

    Example: split_into(32, 3) => [11, 11, 10]
    """
    split = [n / p + 1] * (n % p) + [n / p] * (p - n % p)
    return [int(s) for s in split]


def partition_systems(multi_sys: MultiSystemsVar) -> List[int]:

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


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
