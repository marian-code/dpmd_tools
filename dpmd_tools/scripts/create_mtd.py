from pathlib import Path
import argparse
from typing import List, Optional
import shutil
from ase.io import read, write
import re
from ase.build import make_supercell
import numpy as np
from ssh_utilities import Connection


def input_parser() -> dict:

    p = argparse.ArgumentParser(
        description="Copy, alter and prepare for new run lammps metad directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="choose directory from which will new MetaD dir be created",
    )
    p.add_argument(
        "-o", "--output", type=Path, required=True, help="choose destination directory",
    )
    p.add_argument(
        "-g",
        "--graphs",
        nargs="+",
        type=Path,
        default=None,
        help="input sequence of path pointing to frozen graph files, or glob pattern "
        "that will produce this sequence",
    )
    p.add_argument(
        "-j",
        "--job-name",
        type=str,
        help="job name for PBs script, if not input the one from the template job "
        "will be used",
    )
    p.add_argument(
        "-p", "--press", type=float, help="input target MetaD pressure in GPa"
    )
    p.add_argument(
        "-k",
        "--kelvins",
        type=float,
        dest="temperature",
        help="input target MetaD temperature in K",
    )
    p.add_argument(
        "-s",
        "--structure",
        type=Path,
        dest="instruc",
        help="MetaD starting struture, in not input will use one "
        "from template directory",
    )
    p.add_argument(
        "-gs",
        "--gauss-dep-stride",
        type=int,
        help="set gaussian deposition stride for PLUMMED",
    )
    p.add_argument(
        "-t",
        "--transform",
        type=int,
        nargs="+",
        default=None,
        help="set diagonal for transformation matrix if you want ot make supercell. "
        "e.g. -t 1 2 2 must be three numbers",
    )
    p.add_argument("-w", "--walltime", type=int, help="set walltime for PBS script")

    return vars(p.parse_args())


def create_mtd(
    *,
    input: str,
    output: Path,
    graphs: Optional[List[Path]] = None,
    job_name: Optional[str] = None,
    press: Optional[float] = None,
    temperature: Optional[float] = None,
    instruc: Optional[Path] = None,
    gauss_dep_stride: Optional[int] = None,
    transform: Optional[List[int]] = None,
    walltime: Optional[int] = None,
):

    ignore = [
        "*traj*",
        "log.*",
        "*.txt",
        "*.out",
        "COLVAR",
        "HILLS",
        "bck.*",
        "*.devi",
        "REVIVE.lammps*",
    ]
    if graphs:
        ignore.append("*.pb")

    # copy directory
    if "@" in input:
        server, input = input.split("@")
        print("\n\n\n")
        with Connection(server, local=False, quiet=True) as c:
            c.shutil.download_tree(
                input, output, exclude=ignore, remove_after=False, quiet="stats"
            )
    else:
        shutil.copytree(
            input, output, symlinks=True, ignore=shutil.ignore_patterns(*ignore),
        )

    if graphs:
        # find graphs if input as wildcard
        if len(graphs) == 1:
            graphs = list(Path.cwd().glob(str(graphs[0])))

        graphs = [g.resolve() for g in graphs]

        # check graphs number must be at least 2
        if len(graphs) < 2:
            raise RuntimeError("Must input at least 2 graphs")

        # symlink graphs to target directory
        links = []
        for g in graphs:
            (output / g.name).symlink_to(g)
            links.append(g.name)
    else:
        links = [g.name for g in output.glob("*.pb")]

    # copy in-structure to target dir if it was input,
    # else check if in-structure is present
    if instruc:
        if instruc.suffix == ".instruc":
            atoms = read(instruc, format="lammps-data", style="atomic")
        else:
            atoms = read(instruc)

    elif (output / "data.instruc").is_file():
        atoms = read(output / "data.instruc", format="lammps-data", style="atomic")
    else:
        raise FileNotFoundError(
            f"structure data file {output / 'data.instruc'} is missing"
        )

    # transform cell
    if transform:
        if not len(transform) == 3:
            raise RuntimeError(
                f"Diagonal must have three elements you have input: {transform}"
            )
        prism = np.zeros((3, 3), dtype=int)
        np.fill_diagonal(prism, transform)
        atoms = make_supercell(atoms, prism)

    #  write input structure
    if instruc or (not instruc and transform):
        write(
            output / "data.instruc",
            atoms,
            format="lammps-data",
            force_skew=True,
            atom_style="atomic",
        )

    # guess and alter file based on its contents
    for f in output.glob("*"):
        try:
            text = f.read_text()
        # this is for graph files, and directories
        except (UnicodeDecodeError, IsADirectoryError):
            continue

        # alter lammps input
        if "pair_style" in text:
            print("altering lammps input")
            # set pressure variable in GPa
            if press is not None:
                text = re.sub(
                    r"(variable\s+set_pressure\s+equal\s+)\d+\.?\d+?",
                    r"\g<1>{}".format(press),
                    text,
                )
            # set tepmerature in K
            if temperature is not None:
                text = re.sub(
                    r"(variable\s+temperature_k\s+equal\s+)\d+",
                    r"\g<1>{}".format(temperature),
                    text,
                )
            # set graphs
            if graphs is not None:
                text = re.sub(
                    r"(pair_style\s+deepmd).*?(out_file\s+model.devi)",
                    r"\g<1> {} \g<2>".format(" ".join(links)),
                    text,
                )

        # alter pbs input
        if "PBS" in text or "SBATCH" in text:
            print("altering PBS script")

            if walltime is not None:
                text = re.sub(r"walltime=\d+", f"walltime={walltime:d}", text)

            # find training generation number and NN train order number
            # assumes file name format XXX<gen_num>_<NN_num>XXX
            gen_from_links = [re.findall(r"(\d+)_(\d+)", l)[0] for l in links]

            # get gen numbers and check if all graphs are from same generation
            gens = set([int(g[0]) for g in gen_from_links])
            if len(gens) > 1:
                raise RuntimeError(
                    f"you have input/we have found graphs from multiple "
                    f"generations: {gens}"
                )

            # get train numbers
            numbers = sorted([int(g[1]) for g in gen_from_links])

            if job_name is not None:
                text = re.sub(
                    r"(#PBS\s+-N\s+).*?-\d+_\[\d+-\d+]",
                    r"\g<1>{}-{}_[{}-{}]".format(
                        job_name, gens.pop(), min(numbers), max(numbers)
                    ),
                    text,
                )
            else:
                text = re.sub(
                    r"(#PBS\s+-N\s+.*?)-\d+_\[\d+-\d+]",
                    r"\g<1>-{}_[{}-{}]".format(gens.pop(), min(numbers), max(numbers)),
                    text,
                )

        # alter plummed input
        if "PLUMED" in text:
            print("altering plumed input")
            # set coordination compute atom range
            text = re.sub(
                r"(COORDINATION\s+GROUPA=1-)\d+", r"\g<1>{}".format(len(atoms)), text
            )
            # set appropriate coeffiients for number of atoms
            text = re.sub(
                r"(COEFFICIENTS=)0\.\d+", r"\g<1>{:.6f}".format(1 / len(atoms)), text
            )
            # alter gaussian deposition stride
            if gauss_dep_stride is not None:
                text = re.sub(r"(PACE=)\d+", r"\g<1>{}".format(gauss_dep_stride), text)

        f.write_text(text)


if __name__ == "__main__":

    create_mtd(**input_parser())
