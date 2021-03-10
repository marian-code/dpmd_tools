"""Helper module with dpdata subclasses."""

from .masked_system import MaskedSystem
from typing import Dict, TYPE_CHECKING, Union

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from .clustered_system import ClusteredSystem
    from .selected_system import SelectedSystem

    DEV_DATA = TypedDict(
        "DEV_DATA",
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
            "energies_std": np.ndarray,
            "forces_std": np.ndarray,
        },
    )


class LmpDevSystem(MaskedSystem):
    """Masked System with added energy and forces deviations info.

    This is a special system designed only to read deepmd-lammps trajectories.
    """

    data: "DEV_DATA"
    _additional_arrays = ["energies_std", "forces_std", "used"]
    has_dev_e: bool = True
    has_dev_f: bool = True

    def __new__(cls, *args, **kwargs):
        instance = super(LmpDevSystem, cls).__new__(cls)
        return instance

    def _post_init(self, **kwargs):
        self.data["energies_std"] = kwargs["dev_energy"]
        self.data["forces_std"] = kwargs["dev_force"]


def DevESystem(  # NOSONAR
    data: "DEV_DATA",
    Base: Union["ClusteredSystem", MaskedSystem, "SelectedSystem"],  # NOSONAR
) -> MaskedSystem:
    """Masked/Clustered System with added energy deviations info.

    Notes
    -----
    This is done in such manner that the Base of this class can dynamically be
    assigned as Clustered/Masked system.

    Warnings
    --------
    It is strictly not recommended to instantiate this system it is only ment to be done
    by MaskedSystem class. The constructor is Thus severly limited.
    """

    class DevESystem(Base):
        data: "DEV_DATA"
        _additional_arrays = Base._additional_arrays + ["energies_std"]
        has_dev_e: bool = True

        def __new__(cls, *args, **kwargs):
            instance = super(DevESystem, cls).__new__(cls)
            return instance

    return DevESystem(data=data)


def DevFSystem(  # NOSONAR
    data: "DEV_DATA",
    Base: Union["ClusteredSystem", MaskedSystem, "SelectedSystem"],  # NOSONAR
) -> MaskedSystem:
    """Masked/Clustered System with added force deviations info.

    Notes
    -----
    This is done in such manner that the Base of this class can dynamically be
    assigned as Clustered/Masked system.

    Warnings
    --------
    It is strictly not recommended to instantiate this system it is only ment to be done
    by MaskedSystem class. The constructor is Thus severly limited.
    """

    class DevFSystem(Base):
        data: "DEV_DATA"
        _additional_arrays = Base._additional_arrays + ["forces_std"]
        has_dev_f: bool = True

        def __new__(cls, *args, **kwargs):
            instance = super(DevFSystem, cls).__new__(cls)
            return instance

    return DevFSystem(data=data)


def DevEFSystem(  # NOSONAR
    data: "DEV_DATA",
    Base: Union["ClusteredSystem", MaskedSystem, "SelectedSystem"],  # NOSONAR
) -> MaskedSystem:
    """Masked/Clustered System with added energy and force deviations info.

    Notes
    -----
    This is done in such manner that the Base of this class can dynamically be
    assigned as Clustered/Masked system.

    Warnings
    --------
    It is strictly not recommended to instantiate this system it is only ment to be done
    by MaskedSystem class. The constructor is Thus severly limited.
    """

    class DevEFSystem(Base):
        data: "DEV_DATA"
        _additional_arrays = Base._additional_arrays + ["forces_std", "energies_std"]
        has_dev_e: bool = True
        has_dev_f: bool = True

        def __new__(cls, *args, **kwargs):
            instance = super(DevFSystem, cls).__new__(cls)
            return instance

    return DevEFSystem(data=data)
