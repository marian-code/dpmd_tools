from pathlib import Path
from typing import Tuple
import re
import pandas as pd
import numpy as np


def get_lmp_traj_indices(
    path: Path,
    dev_energy: Tuple[float, float],
    dev_force: Tuple[float, float],
    lammps_infile: str = "in.lammps",
    plumed_file: str = "in.lammps",
) -> np.ndarray:

    header = re.compile(r"#\s*step\s*max_devi_e\s*min_devi_e\s*avg_devi_e\s*max_devi_f")

    #  find deviation file
    for dev_file in path.glob("*"):
        with dev_file.open("r") as stream:
            try:
                if header.match(stream.readline()):
                    break
            except UnicodeDecodeError:  # if we encounter binary graph files
                continue
    else:
        raise FileNotFoundError("Could not find model deviation file from MD run")

    lmp_f = int(
        re.findall(
            r"dump\s+\S+\s+\S+\s+\S+\s+(\d+)", (path / lammps_infile).read_text()
        )[0]
    )
    plm_f = int(re.findall(r"STRIDE=(\d+)", (path / plumed_file).read_text())[0])

    with dev_file.open("r") as f:
        line = f.readline()
        header = re.sub(r"#\s*", "", line).split()
    df = pd.read_table(dev_file, sep=r"\s+", header=0, names=header, comment="#")

    if dev_force:
        cf = (dev_force[0] < df["max_devi_f"]) & (df["max_devi_f"] < dev_force[1])
    else:
        cf = None
    if dev_energy:
        ce = (dev_energy[0] < df["max_devi_f"]) & (df["max_devi_f"] < dev_energy[1])
    else:
        ce = None

    if ce and cf:
        cond = ce & cf
    elif not ce:
        cond = cf
    elif not cf:
        cond = ce

    indices = df.loc[cond]["step"].to_numpy()
    indices = indices.astype(int)

    # compensate for step of plummed and lammps not being equal
    if lmp_f > plm_f:
        indices = indices[indices % lmp_f == 0]
    elif lmp_f < plm_f:
        indices = indices[indices % plm_f == 0]

    # now devide the indices by plummed step so they correspond to lammps trajectory
    indices = indices / plm_f
    indices = indices.astype(int)

    return indices, df
