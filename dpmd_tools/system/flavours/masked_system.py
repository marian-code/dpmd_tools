"""Helper module with dpdata subclasses."""

import os
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from tqdm import tqdm
from typing_extensions import TypedDict

from .base import BaseSystem
from .selected_system import SelectedSystem

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
        },
    )


class AllSelectedError(Exception):
    """Raised when all structures from current dataset have already been selected."""

    pass


class MaskedSystem(BaseSystem):
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
    has_used: bool = True
    _additional_arrays = ["used"]

    def __new__(
        cls,
        *,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional["_DATA"] = None,
        **kwargs
    ):

        if cls.clusters_available(file_name):
            # import here to prevent circular imports
            from .clustered_system import ClusteredSystem
            return ClusteredSystem(
                file_name=file_name,
                fmt=fmt,
                type_map=type_map,
                begin=begin,
                step=step,
                data=data,
                **kwargs
            )
        else:
            instance = super(MaskedSystem, cls).__new__(cls)
            return instance

    @staticmethod
    def clusters_available(file_name: Optional[Path]):
        if file_name is not None:
            if (file_name / "clusters.raw").is_file():
                return True

        return False

    # * custom methods *****************************************************************
    def _post_init(self, **kwargs):
        # check format
        # if reading from raw assume used.raw is already in-place
        if "raw" in kwargs["fmt"]:

            if not kwargs["file_name"]:
                raise ValueError("must pass in filename")
            used_file: Path = kwargs["file_name"] / "used.raw"

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

            # when user forces some iteration as default, slice the used array
            # accordingly and save a backup file
            if kwargs["force_iteration"] is not None:
                fi = kwargs["force_iteration"]
                if fi > 0:
                    fi += 1  # need to shift because np slicing will take n-1

                self.load_messages.append(
                    f" - forced iteration to {fi} from {self.data['used'].shape[1]}"
                )
                backup_used = used_file.with_suffix(".raw.backup")
                if backup_used.is_file():
                    backup_used.unlink()
                copy2(used_file, backup_used)
                self.load_messages.append(
                    f"saved backup file {backup_used}"
                )
                self.data["used"] = self.data["used"][:, fi]


        # else init from scratch but not if data with 'used' key are passed in
        else:
            try:
                self.data["used"]  # type: ignore
            except (KeyError, TypeError):
                self.data["used"] = np.zeros(
                    (self.data["cells"].shape[0], 1), dtype=int
                )

        # cond > 1 is for predict when we generate subsystems of length == 1
        # check for data presence avoids fail on copy and sybsystem creation
        if (
            np.count_nonzero(self.mask) >= len(self.data["cells"])
            and len(self.mask) > 1
            and not kwargs["data"]
        ):
            raise AllSelectedError("All structures are alread selected")

    @property
    def iteration(self) -> int:
        """Return curent selection iteration."""
        return self.data["used"].shape[1]

    def append_mask(self, selected_indices: np.ndarray):
        """Add mask column resulting from new selection iteration."""
        # create used structures mask for cuttent iteration
        mask = np.zeros(self.get_nframes(), dtype=int)
        mask[selected_indices] = 1

        if self.data["used"].shape[1] == 1 and self.data["used"].sum() == 0:
            self.data["used"] = np.atleast_2d(mask).T
        else:
            self.data["used"] = np.hstack((self.data["used"], np.atleast_2d(mask).T))

    def get_subsystem_indices(self, iteration: Optional[int] = None) -> np.ndarray:
        """Get indices of frames filtered by previous iterations.

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

    def get_subsystem(self, iteration: Optional[int] = None) -> "SelectedSystem":
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
        sub_system : MaskedSystem
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
        #  get subsystem indices for specified iteration
        f_idx = self.get_subsystem_indices(iteration)

        # get subsystem data
        tmp_data = super(MaskedSystem, self).sub_system(f_idx).data

        # now add the iteration data to subsystem data
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
        tmp_data["iteration"] = iteration_indices[frame_indices.argsort()]

        # initialize Selected system class with subsystem data and return
        # TODO ugly hack
        if self.has_dev_e and self.has_dev_f:
            from .dev_system import DevEFSystem
            return DevEFSystem(tmp_data, SelectedSystem)  # type: ignore
        elif self.has_dev_e:
            from .dev_system import DevESystem
            return DevESystem(tmp_data, SelectedSystem)  # type: ignore
        elif self.has_dev_f:
            from .dev_system import DevFSystem
            return DevFSystem(tmp_data, SelectedSystem)  # type: ignore
        else:
            return SelectedSystem(data=tmp_data)

    def add_dev_e(self, data: np.ndarray) -> "MaskedSystem":
        # TODO ugly hack
        self.data["energies_std"] = data
        self._additional_arrays.append("energies_std")
        self.has_dev_e = True
        from .dev_system import DevESystem
        return DevESystem(self.data, type(self))  # type: ignore

    def add_dev_f(self, data: np.ndarray) -> "MaskedSystem":
        # TODO ugly hack
        self.data["forces_std"] = data
        self._additional_arrays.append("forces_std")
        self.has_dev_f = True
        from .dev_system import DevFSystem
        return DevFSystem(self.data, type(self))  # type: ignore

    # * overriding methods *************************************************************
    def to_deepmd_raw(self, folder: Path, append: bool):
        """Output to raw data format.

        Parameters
        ----------
        folder : Path
            destination directory
        append : bool
            if True, than only `used.raw` is updated other files are left untouched
        """
        if not append:
            super(MaskedSystem, self).to_deepmd_raw(folder)
        else:
            np.savetxt(folder / "used.raw", self.data["used"], fmt="%d")

    def predict(self, dp: Path):
        """Predict energies and forces by deepmd-kit for yet unselected structures.

        Parameters
        ----------
        dp : deepmd.DeepPot or str
            The deepmd-kit potential class or the filename of the model.

        Returns
        -------
        labeled_sys MaskedSystem
            The labeled system.
        """
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        import deepmd.DeepPot as DeepPot

        self.deeppot = DeepPot(str(dp))
        type_map = self.deeppot.get_type_map()
        ori_sys = self.copy()
        ori_sys.sort_atom_names(type_map=type_map)
        atype = ori_sys["atom_types"]

        labeled_sys = self.copy()

        # these were already selected in previous iterations there is no need to
        # compute for these indices since they wont be selected again anyway
        dont_recompute_idx = self.get_subsystem_indices()

        for idx in tqdm(
            range(self.get_nframes()), total=self.get_nframes(), ncols=100, leave=False
        ):

            ss = super(MaskedSystem, self).sub_system(idx)

            coord = ss["coords"].reshape((-1, 1))
            if not ss.nopbc:
                cell = ss["cells"].reshape((-1, 1))
            else:
                cell = None

            data = ss.data

            if idx in dont_recompute_idx:
                # already selected in previous iteration -> skip
                # create fake data, it does not matter since these will be skipped
                # anyway because of the mask
                data["energies"] = np.zeros((1, 1))
                data["forces"] = np.zeros((1, ss.get_natoms(), 3))
                data["virials"] = np.zeros((1, 3, 3))
            else:
                e, f, v = self.deeppot.eval(coord, cell, atype)
                data["energies"] = e.reshape((1, 1))
                data["forces"] = f.reshape((1, -1, 3))
                data["virials"] = v.reshape((1, 3, 3))

            labeled_sys.data["energies"][idx] = data["energies"]
            labeled_sys.data["forces"][idx] = data["forces"]
            labeled_sys.data["virials"][idx] = data["virials"]
        return labeled_sys
