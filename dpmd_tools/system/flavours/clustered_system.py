"""Helper module with dpdata subclasses."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from typing_extensions import TypedDict

from .masked_system import MaskedSystem
from .base import BaseSystem

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
            "used": np.ndarray,
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
    _additional_arrays = ["used", "clusters"]

    def _post_init(self, **kwargs):
        super(ClusteredSystem, self)._post_init(**kwargs)
        if not kwargs["data"]:
            self.data["clusters"] = np.loadtxt(kwargs["file_name"] / "clusters.raw")

    @property
    def clusters(self) -> Optional[np.ndarray]:
        """Get cluster index for all structures."""
        return self.data["clusters"]
