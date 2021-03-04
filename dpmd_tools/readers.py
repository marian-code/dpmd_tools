import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, List, Optional

from dpdata import LabeledSystem

from dpmd_tools.system import MaskedSystem

if TYPE_CHECKING:
    from ase import Atoms


def load_npy_data(path: Path) -> List["Atoms"]:

    system = LabeledSystem()

    system.from_deepmd_comp(str(path.resolve()))
    return system.to_ase_structure()


def load_raw_data(path: Path) -> List["Atoms"]:
    return LabeledSystem(str(path), fmt="deepmd/raw").to_ase_structure()


def _extract(archive: Path, to: Path) -> Optional[Path]:
    extract_to = to / archive.stem

    try:
        with gzip.open(archive, "rb") as infile:
            with extract_to.open("wb") as outfile:
                shutil.copyfileobj(infile, outfile)
    except Exception as e:
        print(e)
        return None
    else:
        return extract_to


def read_xtalopt_dir(path: Path) -> List[MaskedSystem]:

    systems = []
    with TemporaryDirectory(dir=path, prefix="temp_") as tmp:
        tempdir = Path(tmp)
        for p in path.glob("OUTCAR.*.gz"):
            if p.suffixes[0] in (".1", ".2"):
                print(f"skipping file {p.name}")
                continue
            outcar = _extract(p, tempdir)

            if outcar:
                systems.append(MaskedSystem(outcar, fmt="vasp/outcar"))

    return systems


def read_vasp_dir(path: Path) -> List[MaskedSystem]:

    outcar = path / "OUTCAR"
    return read_vasp_out(outcar)


def read_vasp_out(outcar: Path) -> List[MaskedSystem]:
    return [MaskedSystem(outcar, fmt="vasp/outcar")]


def read_dpmd_raw(system_dir: Path) -> List[MaskedSystem]:
    return [MaskedSystem(system_dir, fmt="deepmd/raw")]
