from pathlib import Path
from typing import TYPE_CHECKING, List

from dpdata import LabeledSystem

if TYPE_CHECKING:
    from ase import Atoms


def load_npy_data(path: Path) -> List["Atoms"]:

    system = LabeledSystem()

    system.from_deepmd_comp(str(path.resolve()))
    return system.to_ase_structure()


def load_raw_data(path: Path) -> List["Atoms"]:
    return LabeledSystem(str(path), fmt="deepmd/raw").to_ase_structure()
