"""Helper module with dpdata subclasses."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from .base import BaseSystem
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


class SelectedSystem(BaseSystem):
    """System of selected structures for training from all selection iterations.

    Has additional data field `iteration` which tells in what iteration was the
    respective structure selected.

    The main purpose of this class is to export trining subsystem correctly
    """

    data: "SEL_DATA"
    has_iterations: bool = True
    _additional_arrays = ["iteration"]

    @property
    def iteration(self):
        return self.data["iteration"].max()

    def to_deepmd_npy(self, folder: Path, set_size: int = 5000, prec: Any = np.float32):
        super(SelectedSystem, self).to_deepmd_npy(
            folder, set_size=set_size, prec=prec
        )
        np.savetxt(folder / "iteration.raw", self.data["iteration"], fmt="%d")
