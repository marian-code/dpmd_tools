"""Helper module with dpdata subclasses."""

from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union, List
from pathlib import Path

import numpy as np
from typing_extensions import TypedDict

from .masked_system import MaskedSystem

if TYPE_CHECKING:

    CLUST_DATA = TypedDict(
        "CLUST_DATA",
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
            "clusters": np.ndarray,
        },
    )


class ClusteredSystem(MaskedSystem):
    """Maskedystem with added clusters info.

    Warnings
    --------
    Do not instantiate! use MaskedSystem class which will output this class if it finds
    `clusters.raw` file
    """

    data: "CLUST_DATA"
    has_clusters: bool = True

    def __new__(cls, *args, **kwargs):
        instance = super(ClusteredSystem, cls).__new__(cls)
        return instance

    def __init__(  # NOSONAR
        self,
        *,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional["CLUST_DATA"] = None,
        **kwargs,
    ) -> None:
        super(ClusteredSystem, self).__init__(
            file_name=file_name,
            fmt=fmt,
            type_map=type_map,
            begin=begin,
            step=step,
            data=data,
            **kwargs,
        )

        if not data:
            self.data["clusters"] = np.loadtxt(file_name / "clusters.raw")

    @property
    def clusters(self) -> Optional[np.ndarray]:
        """Get cluster index for all structures."""
        return self.data["clusters"]

    def copy(self):
        tmp_sys = super(ClusteredSystem, self).copy()
        tmp_sys.data["clusters"] = deepcopy(self.data["clusters"])
        return tmp_sys

    def append(self, system: "ClusteredSystem"):

        super(ClusteredSystem, self).append(system)

        if not isinstance(system, ClusteredSystem):
            raise TypeError(
                f"The appending system is of wrong type, expected: "
                f"ClusteredSystem, got {type(system)}"
            )
        else:
            self.data["clusters"] = np.concatenate(
                (self.data["clusters"], system.data["clusters"]), axis=0
            )

    def sub_system(self, f_idx: Union[np.ndarray, int]) -> "ClusteredSystem":
        tmp_sys = super(ClusteredSystem, self).sub_system(f_idx)
        tmp_sys.data["clusters"] = self.data["clusters"][f_idx]
        if isinstance(f_idx, int):
            tmp_sys.data["clusters"] = np.atleast_2d(tmp_sys.data["clusters"])

        return ClusteredSystem(data=tmp_sys.data)
