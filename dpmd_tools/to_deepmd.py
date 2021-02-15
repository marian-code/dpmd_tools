"""Read various data formats to deepmd.

Can be easily extended by writing new parser functions like e.g.
`read_vasp_out` and by altering get paths function.
"""

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["KMP_WARNINGS"] = "FALSE"
import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, Any
from warnings import warn
import plotly.graph_objects as go

import numpy as np
from ase.atoms import Atoms
from dpmd_tools.data import MultiSystemsVar
from dpdata import LabeledSystem
from tqdm import tqdm
import pandas as pd

WORK_DIR = Path.cwd()
DPMD_DATA = WORK_DIR / "deepmd_data"
MIN_STRUCTURES = 30
PARSER_CHOICES = ("xtalopt", "vasp_dirs", "vasp_files", "dpmd_raw")


def input_parser():
    p = argparse.ArgumentParser(
        description="load vasp OUTCARs from subdirectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("-p", "--parser", default=None, required=True, type=str,
                   choices=PARSER_CHOICES, help="input parser you wish to use")
    p.add_argument("-g", "--graphs", default=[], type=str, nargs="*",
                   help="input list of graphs you wish to use for checking if "
                   "datapoint is covered by current model. If present must "
                   "have at least two distinct graphs")
    p.add_argument("-e", "--every", default=None, type=int, help="take every "
                   "n-th frame")
    p.add_argument("-v", "--volume", default=None, type=str, nargs=2,
                   help="constrain structures volume. Input as 10.0 31. "
                   "In [A^3]")
    p.add_argument("-n", "--energy", default=None, type=str, nargs=2,
                   help="constrain structures energy. Input as -5 -2. In [eV]")
    p.add_argument("-a", "--per-atom", default=False, action="store_true",
                   help="set if energy and volume constraints are computed "
                   "per atom or for the whole structure")
    p.add_argument("-gp", "--get-paths", default=None, type=str, help="if not "
                   "specified default function will be used. Otherwise you "
                   "can input python code as string that outputs list of "
                   "'Path' objects to paths variable. The path object is "
                   "already imported for you. Example: "
                   "-g 'paths = [Path.cwd() / \"OUTCAR\"]'")

    args = vars(p.parse_args())
    if len(set(args["graphs"])) == 1:
        raise ValueError("If you want to filter structures based on current "
                         "graphs you must input at least two")

    return args


class Loglprint:

    def __init__(self, log_file: Path) -> None:
        self.log_stream = log_file.open("w")

    def __call__(self, msg: Any) -> Any:
        self.log_stream.write(f"{str(msg)}\n")
        print(msg)


def extract(archive: Path, to: Path) -> Optional[Path]:
    extract_to = to / archive.stem

    try:
        with gzip.open(archive, "rb") as infile:
            with extract_to.open("wb") as outfile:
                shutil.copyfileobj(infile, outfile)
    except Exception as e:
        lprint(e)
        return None
    else:
        return extract_to


def read_xtalopt_dir(path: Path) -> List[LabeledSystem]:

    systems = []
    with TemporaryDirectory(dir=path, prefix="temp_") as tmp:
        tempdir = Path(tmp)
        for p in path.glob("OUTCAR.*.gz"):
            if p.suffixes[0] in (".1", ".2"):
                continue
            outcar = extract(p, tempdir)

            if outcar:
                systems.append(LabeledSystem(str(outcar.resolve()),
                                             fmt="vasp/outcar"))

    return systems


def read_vasp_dir(path: Path) -> List[LabeledSystem]:

    outcar = path / "OUTCAR"
    return read_vasp_out(outcar)


def read_vasp_out(outcar: Path) -> List[LabeledSystem]:
    return [LabeledSystem(str(outcar.resolve()), fmt="vasp/outcar")]


def read_dpmd_raw(system_dir: Path) -> List[LabeledSystem]:
    return [LabeledSystem(str(system_dir.resolve()), fmt="deepmd/raw")]


def get_paths() -> List[Path]:

    # paths = [d for d in WORK_DIR.glob("*/") if d.is_dir()]

    """
    paths = []
    for root, dirs, files in os.walk(WORK_DIR, topdown=True):

        for f in files:
            if "OUTCAR" in f:
                paths.append(Path(root) / f)

        delete = []
        for i, d in enumerate(dirs):
            if "analysis" in d or "traj" in d or "results" in d or "test" in d:
                delete.append(i)

        for i in sorted(delete, reverse=True):
            del dirs[i]
    """

    paths = [WORK_DIR]

    return paths


def plot(multi_sys: MultiSystemsVar):

    dataframes = []
    for system in multi_sys:
        n = system.get_natoms()
        dataframes.append(pd.DataFrame({
            "energies": system.data["energies"].flatten() / n,
            "volumes": np.linalg.det(system.data["cells"]) / n,
            "energies_std": system.data["energies_std"].flatten(),
            "forces_std_max": system.data["forces_std_max"]
        }))

    data = pd.concat(dataframes)

    for what in ("energies_std", "forces_std_max"):
        fig = go.Figure(data=go.Scattergl(
            x=data["energies"],
            y=data["volumes"],
            mode='markers',
            marker=dict(
                size=8,
                color=data[what],
                colorbar=dict(
                    title=what
                ),
                colorscale='thermal',
                showscale=True
            )
        ))
        fig.update_layout(
            title='E-V plot',
            xaxis_title="V [A^3]",
            yaxis_title="E [eV]",
            coloraxis_colorbar=dict(
                title=what,
            )
        )
        fig.write_html(f"{what}.html", include_plotlyjs="cdn")


class ApplyConstraint:
    """Apply constraints to dataset and filter out unsatisfactory structures.

    Each method must concatenate to `del_indices` the indices of structures it
    found that do not satisfy the imposed conditions.
    """

    _predictions = None
    del_indices: np.ndarray = np.empty((0, ), dtype=int)

    def __init__(self, name: str, system: LabeledSystem) -> None:

        lprint(f"filtering system {name}")
        self.system: LabeledSystem = system
        self.atoms: List[Atoms] = system.to_ase_structure()

    def get_predictions(self, graphs: List[str]) -> List[LabeledSystem]:

        if not self._predictions:
            lprint("computing model predictions")
            self._predictions = []
            job = tqdm(enumerate(graphs, 1), ncols=100, total=len(graphs))
            for i, g in job:
                job.set_description(f"graph {i}/{len(graphs)}")
                self._predictions.append(self.system.predict(g))
        return self._predictions

    def energy(self, *, bracket: Tuple[float, float], per_atom: bool = True):
        lprint(f"based on energy{' per atom' if per_atom else ''}")

        energies = self.system.data["energies"]
        if per_atom:
            energies = energies.copy() / self.system.get_natoms()

        d = np.argwhere((energies < bracket[0]) | (energies > bracket[1]))

        lprint(f"got {len(d)} entries to delete as a result of energy "
               f"constraints")

        self.del_indices = np.concatenate((self.del_indices, d.flatten()))

    def every(self, *, n_th: int):
        lprint(f"based on: take every {n_th} frame criterion")

        d = np.delete(
            np.arange(self.system.get_nframes()),
            np.arange(0, self.system.get_nframes(), n_th)
        )

        lprint(f"got {len(d)} entries to delete as a result of take every n-th"
               f" frame constraints")

        self.del_indices = np.concatenate((self.del_indices, d))

    def volume(self, *, bracket: Tuple[float, float], per_atom: bool = True):
        lprint(f"based on volume{' per atom' if per_atom else ''}")

        # this an array of square matrices, linalg det auto cancualtes det for
        # every sub matrix that is square
        volumes = np.linalg.det(self.system.data["cells"])
        if per_atom:
            volumes /= self.system.get_natoms()

        d = np.argwhere((volumes < bracket[0]) | (volumes > bracket[1]))

        lprint(f"got {len(d)} entries to delete as a result of volume "
               f"constraints")

        self.del_indices = np.concatenate((self.del_indices, d.flatten()))

    def dev_e(self, *, graphs: List[str], max_dev_e: float,
              std_method: bool = False):
        """Select which labeled structures should be added to dataset.

        This method is usefull when you already have DFT data labels without
        any prior selection and want to decide which of them should be added to
        the dataset based on dataset predictions for them.

        See Also
        --------
        :func:`dpgen.simplify.simplify.post_model_devi`
        """
        lprint("based on energy std")
        predictions = self.get_predictions(graphs)

        # shape: (n_models, n_frames)
        energies = np.stack([p.data["energies"] for p in predictions])
        energies /= self.system.get_natoms()

        if std_method:
            e_std = energies.std(axis=0)
        else:
            reference = self.system.data["energies"] / self.system.get_natoms()
            # make column vector of reference DFT data
            reference = np.atleast_2d(reference).T
            e_std = np.sqrt(np.mean(np.power(abs(energies - reference), 2),
                            axis=0))

        # save for plotting
        self.system.data["energies_std"] = e_std

        d = np.argwhere(e_std < max_dev_e)

        lprint(f"got {len(d)} entries to delete as a result of energy std "
               f"constraints")

        self.del_indices = np.concatenate((self.del_indices, d.flatten()))

    def dev_f(self, *, graphs: List[str], max_dev_f: float,
              std_method: bool = False):
        """Select which labeled structures should be added to dataset.

        This method is usefull when you already have DFT data labels without
        any prior selection and want to decide which of them should be added to
        the dataset based on dataset predictions for them.

        See Also
        --------
        :func:`dpgen.simplify.simplify.post_model_devi`
        """
        lprint(f"based on max atom force std")
        predictions = self.get_predictions(graphs)

        # shape: (n_models, n_frames, n_atoms, 3)
        forces = np.stack([p.data["forces"] for p in predictions])

        # shape: (n_frames, n_atoms, 3)
        if std_method:
            f_std = np.std(axis=0)
        else:
            reference = self.system.data["forces"]
            f_std = np.sqrt(np.mean(np.power(abs(forces - reference), 2),
                            axis=0))

        # shape: (n_fames, n_atoms)
        f_std_size = np.linalg.norm(f_std, axis=2)

        # shape: (n_frames, )
        f_std_max = np.max(f_std_size, axis=1)

        # save for plotting
        self.system.data["forces_std_max"] = f_std_max

        d = np.argwhere(f_std_max < max_dev_f)

        lprint(f"got {len(d)} entries to delete as a result of max forces std "
               f"constraints")

        self.del_indices = np.concatenate((self.del_indices, d.flatten()))

    def apply(self) -> LabeledSystem:
        lprint(f"deleting {len(self.del_indices)} entries")
        for attr in ['cells', 'coords', 'energies', 'forces', 'virials']:
            system.data[attr] = np.delete(system.data[attr], self.del_indices,
                                          axis=0)

        return system


if __name__ == "__main__":

    args = input_parser()
    lprint = Loglprint(DPMD_DATA / "README")

    if args["get_paths"]:
        exec(args["get_paths"])
        try:
            paths
        except NameError:
            raise RuntimeError("your script did not assign to 'paths' "
                               "variable")
    else:
        paths = get_paths()

    lprint("will read from this paths:")
    for p in paths:
        lprint(f"-{p}")

    if not args["graphs"]:
        warn("It is strongly advised to use filtering based on currently "
             "trained model", UserWarning)

    multi_sys = MultiSystemsVar()

    if args["parser"] == "xtalopt":
        multi_sys.collect_cf(paths, read_xtalopt_dir)
    elif args["parser"] == "vasp_dirs":
        multi_sys.collect_cf(paths, read_vasp_dir)
    elif args["parser"] == "vasp_files":
        multi_sys.collect_cf(paths, read_vasp_out)
    elif args["parser"] == "dpmd_raw":
        multi_sys.collect_cf(paths, read_dpmd_raw)
    else:
        raise NotImplementedError(f"parser for {args['parser']} "
                                  f"is not implemented")

    lprint("got these systems -----------------------------------------------")
    for name, system in multi_sys.items():
        lprint(f"{name:6} -> {len(system):4} structures")

    lprint("filtering data --------------------------------------------------")

    multi_sys.predict(args["graphs"])

    lprint(f"size before {sum([len(s) for s in multi_sys.values()])}")
    for k, system in multi_sys.items():
        # for k, system in [("Ge8", multi_sys.systems["Ge8"])]:
        constraints = ApplyConstraint(k, system)
        if args["energy"]:
            bracket = [float(e) for e in args["energy"]]
            constraints.energy(bracket=bracket, per_atom=args["per_atom"])
        if args["volume"]:
            bracket = [float(v) for v in args["volume"]]
            constraints.volume(bracket=bracket, per_atom=args["per_atom"])
        if args["graphs"]:
            constraints.dev_e(graphs=args["graphs"], max_dev_e=1e-2)
            constraints.dev_f(graphs=args["graphs"], max_dev_f=1e-2)
        if args["every"]:
            constraints.every(n_th=int(args["every"]))
        system = constraints.apply()
        lprint("*************************************************************")

    lprint(f"size after {sum([len(s) for s in multi_sys.values()])}")

    if args["graphs"]:
        lprint("plotting std for energies and max atom forces")
        plot(multi_sys)

    lprint(f"deleting systems with less than {MIN_STRUCTURES} structures ----")
    del_systems = []
    for name, system in multi_sys.items():
        if len(system) < MIN_STRUCTURES:
            del_systems.append(name)

    for s in del_systems:
        lprint(f"deleting {s}")
        multi_sys.systems.pop(s, None)

    lprint("shuffling systems -----------------------------------------------")
    multi_sys.shuffle()

    lprint("saving data -----------------------------------------------------")
    multi_sys.to_deepmd_raw(DPMD_DATA)
    multi_sys.to_deepmd_npy(DPMD_DATA, set_size="auto")
