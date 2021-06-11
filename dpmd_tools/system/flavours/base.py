"""Helper module with dpdata subclasses."""

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
from dpdata import LabeledSystem
from typing_extensions import TypedDict

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
        },
    )


class BaseSystem(LabeledSystem):
    """Base class for further subclassing, houses all the common methods."""

    data: "_DATA"
    _additional_arrays: List[str] = []
    # append mesages from class initialization when reading data and they will be
    # printed after data has completed loading
    load_messages: List[str] = []

    def __init__(  # NOSONAR
        self,
        *,
        file_name: Optional[Path] = None,
        fmt: str = "auto",
        type_map: List[str] = None,
        begin: int = 0,
        step: int = 1,
        data: Optional["_DATA"] = None,
        **kwargs,
    ) -> None:
        super(BaseSystem, self).__init__(
            file_name=str(file_name) if file_name else None,
            fmt=fmt,
            type_map=type_map,
            begin=begin,
            step=step,
            data=data,
            **kwargs,
        )
        kwargs.update(
            dict(
                file_name=file_name,
                fmt=fmt,
                type_map=type_map,
                begin=begin,
                step=step,
                data=data
            )
        )
        self._post_init(**kwargs)

    # * custom methods *****************************************************************
    def _post_init(self, **kwargs):
        """Override in subclass to define custom behaviour."""
        pass

    def __getattr__(self, attr: str) -> bool:
        """Default behaviour for has_<attribute> is false.

        Override in subclass by: attr=True
        """
        if "has_" in attr:
            return False
        else:
            raise AttributeError(f"{type(self)} does not have attribute {attr}")

    # * overriding methods *************************************************************
    def to_deepmd_npy(self, folder: Path, set_size: int = 5000, prec: Any = np.float32):
        super(BaseSystem, self).to_deepmd_npy(
            str(folder), set_size=set_size, prec=prec
        )

    def to_deepmd_raw(self, folder: Path):
        """Output to raw data format.

        Parameters
        ----------
        folder : Path
            destination directory
        """
        super(BaseSystem, self).to_deepmd_raw(str(folder))
        for a in self._additional_arrays:
            np.savetxt(folder / f"{a}.raw", self.data[a], fmt="%d")

    def copy(self):
        """Manipulates also arrays defined by additional_arrays beyond standard set."""
        tmp_sys = super(BaseSystem, self).copy()
        for a in self._additional_arrays:
            tmp_sys.data[a] = deepcopy(self.data[a])
        return tmp_sys

    def append(self, system: "BaseSystem"):
        """Manipulates also arrays defined by additional_arrays beyond standard set."""
        if not isinstance(system, type(self)):
            raise TypeError(
                f"The appending system is of wrong type, expected: "
                f"{type(self)}, got {type(system)}"
            )
        else:
            super(BaseSystem, self).append(system)
            for a in self._additional_arrays:
                self.data[a] = np.concatenate((self.data[a], system[a]), axis=0)

    def sub_system(self, f_idx: Union[np.ndarray, int, slice]) -> "BaseSystem":
        """Manipulates also arrays defined by additional_arrays attribute.
        
        The standard set is manimpulated by dpdata.
        """
        tmp_sys = super(BaseSystem, self).sub_system(f_idx)
        for a in self._additional_arrays:
            tmp_sys.data[a] = self.data[a][f_idx]
            if isinstance(f_idx, int):
                tmp_sys.data[a] = np.atleast_2d(tmp_sys.data[a])

        return type(self)(data=tmp_sys.data)

    def shuffle(self):
        """Manipulates also arrays defined by additional_arrays beyond standard set."""
        idx = super(BaseSystem, self).shuffle()
        for a in self._additional_arrays:
            self.data[a] = self.data[a][idx]
        return idx

    def __str__(self):

        s = "NAME" + " " * 16 + ": SHAPE\n"
        for k, v in self.data.items():
            try:
                s += f"{k:20}: {v.shape}\n"
            except AttributeError:
                try:
                    s += f"{k:20}: {len(v)}\n"
                except Exception:
                    s += f"{k:20}: {v}\n"



        return s