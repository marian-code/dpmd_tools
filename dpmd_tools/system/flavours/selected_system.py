"""Helper module with dpdata subclasses."""

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
from dpdata import LabeledSystem
from typing_extensions import TypedDict

if TYPE_CHECKING:

    SEL_DATA = TypedDict(
        "SEL_DATA",
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
            "iteration": np.ndarray,
        },
    )


class SelectedSystem(LabeledSystem):
    """System of selected structures for training from all selection iterations.

    Has additional data field `iteration` which tells in what iteration was the
    respective structure selected.

    The main purpose of this class is to export trining subsystem correctly
    """

    data: "SEL_DATA"

    def __init__(
        self,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional["SEL_DATA"] = None,
        **kwargs,
    ) -> None:
        super(SelectedSystem, self).__init__(
            file_name=str(file_name) if file_name else None,
            fmt=fmt,
            type_map=type_map,
            begin=begin,
            step=step,
            data=data,
            **kwargs,
        )

    def append(self, system: "SelectedSystem"):
        super(SelectedSystem, self).append(system)

        if not isinstance(system, SelectedSystem):
            raise TypeError(
                f"The appending system is of wrong type, expected: "
                f"SelectedSystem, got {type(system)}"
            )
        else:
            self.data["iteration"] = np.concatenate(
                (self.data["iteration"], system.data["iteration"]), axis=0
            )

    def shuffle(self):
        """Also shuffle labeled data e.g. energies and forces."""
        idx = super(SelectedSystem, self).shuffle()
        self.data["iteration"] = self.data["iteration"][idx]
        return idx

    @property
    def iteration(self):
        return self.data["iteration"].max()

    def to_deepmd_npy(self, folder: Path, set_size: int = 5000, prec: Any = np.float32):
        super(SelectedSystem, self).to_deepmd_npy(
            str(folder), set_size=set_size, prec=prec
        )
        np.savetxt(folder / "iteration.raw", self.data["iteration"], fmt="%d")

    def to_deepmd_raw(self, folder: Path):
        super(SelectedSystem, self).to_deepmd_raw(str(folder))
        np.savetxt(folder / "iteration.raw", self.data["iteration"], fmt="%d")

    def copy(self):
        tmp_sys = super(SelectedSystem, self).copy()
        tmp_sys.data["iteration"] = deepcopy(self.data["iteration"])
        return tmp_sys

    def sub_system(self, f_idx: Union[np.ndarray, int]) -> "SelectedSystem":
        tmp_sys = super(SelectedSystem, self).sub_system(f_idx)
        tmp_sys.data["iteration"] = self.data["iteration"][f_idx]
        if isinstance(f_idx, int):
            tmp_sys.data["iteration"] = np.atleast_2d(tmp_sys.data["iteration"])

        return SelectedSystem(data=tmp_sys.data)

    def predict(self, dp: Path):
        raise NotImplementedError("Selected system is only for exporting")
