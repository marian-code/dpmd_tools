from dpmd_tools.readers.to_ase import load_dev_lmp_traj
from typing import Optional, Tuple, TYPE_CHECKING
from ase.io.extxyz import write_xyz
from colorama import Fore

if TYPE_CHECKING:
    from pathlib import Path


def dev2ase(
    *,
    trajectory_file: "Path",
    dev_energy: Tuple[float, float],
    dev_force: Tuple[float, float],
    portion: Optional[float],
    nframes: Optional[int],
    lammps_infile: str,
    plumed_infile: str,
    **kwargs,
):
    atoms = load_dev_lmp_traj(
        trajectory_file,
        dev_energy,
        dev_force,
        portion,
        nframes,
        lammps_infile,
        plumed_infile,
    )

    print(f"{Fore.RED}All atom types will be changed to Ge !!!")

    for a in atoms:
        a.set_chemical_symbols(["Ge"] * len(a))

    write_xyz(
        f"{trajectory_file.resolve().parent.name}_recompute.xyz",
        atoms,
        write_info=True,
        columns=["symbols", "positions", "numbers"],
    )

