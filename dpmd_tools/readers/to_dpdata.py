"""Helper reader function for various data types.

Warnings
--------
Obey naming convetion `read_<sometype>` because `to_deepmd` script relies on it!
Alwys append new methods to __all__ variable!
"""

import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, List, Optional, Tuple

import numpy as np
from dpmd_tools.system import LmpDevSystem, MaskedSystem, System
from .common import get_lmp_traj_indices

__all__ = [
    "read_xtalopt_dir",
    "read_vasp_dir",
    "read_vasp_file",
    "read_dpmd_raw",
    "read_dpmd_npy",
    "read_lmp_traj_dev",
    "read_outcar_relax",
]


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


def read_xtalopt_dir(path: Path, **kwargs) -> List[MaskedSystem]:

    systems = []
    with TemporaryDirectory(dir=path, prefix="temp_") as tmp:
        tempdir = Path(tmp)
        for p in path.glob("OUTCAR.*.gz"):
            if p.suffixes[0] in (".1", ".2"):
                print(f"skipping file {p.name}")
                continue
            outcar = _extract(p, tempdir)

            if outcar:
                systems.append(System(file_name=outcar, fmt="vasp/outcar"))

    return systems


def read_vasp_dir(path: Path, **kwargs) -> List[MaskedSystem]:

    outcar = path / "OUTCAR"
    return read_vasp_file(outcar)


def read_vasp_file(outcar: Path, **kwargs) -> List[MaskedSystem]:
    return [System(file_name=outcar, fmt="vasp/outcar")]



def read_outcar_relax(outcar: Path, **kwargs) -> List[MaskedSystem]:
    system = System(file_name=outcar, fmt="vasp/outcar")
    return system.sub_system(-1)


def read_dpmd_raw(
    system_dir: Path, force_iteration: Optional[int] = None, **kwargs
) -> List[MaskedSystem]:
    return [
        System(
            file_name=system_dir, fmt="deepmd/raw", force_iteration=force_iteration
        )
    ]

def read_dpmd_npy(
    system_dir: Path, force_iteration: Optional[int] = None, **kwargs
) -> List[MaskedSystem]:
    return [
        System(
            file_name=system_dir, fmt="deepmd/npy", force_iteration=force_iteration
        )
    ]


def read_lmp_traj_dev(
    traj_file: Path,
    dev_energy: Tuple[float, float],
    dev_force: Tuple[float, float],
    **kwargs,
) -> List[LmpDevSystem]:
    from py_extract_frames.extract_frames import copy_traj, parse_traj

    with TemporaryDirectory(dir=traj_file.parent) as tmp:
        tmp_dir = Path(tmp)

        indices, dev_data = get_lmp_traj_indices(tmp_dir, dev_energy, dev_force)

        systems = []
        for start, stop in _consecutive(indices):
            # now cut only the needed part of the trajectory
            # parse_traj takes the last index as oposed to python slicing which
            # takes stop-1 as last
            start_pos, stop_pos = parse_traj(traj_file, from_=start, to=stop - 1)
            copy_traj(traj_file, tmp_dir / "traj.tmp", start_pos, stop_pos)
            s = LmpDevSystem(
                file_name=tmp_dir / "traj.tmp",
                dev_energy=dev_data["max_devi_e"].to_numpy()[start:stop],
                dev_force=dev_data["max_devi_f"].to_numpy()[start:stop],
            )
            systems.append(s)

        return systems


def _consecutive(data: np.ndarray, stepsize: int = 1) -> Iterator[Tuple[int, int]]:
    #  data=[1 2 3 8 9 10 15 16]
    data.sort()
    # indices=[0 2 5 8]
    indices = (
        [0]
        + (np.argwhere(np.diff(data) > stepsize).flatten() + 1).tolist()  # [2 5]
        + [len(data)]  # [8]
    )
    # zip=[(0 2), (2, 5), (5, 8)]
    return list(zip(indices, indices[1:]))
