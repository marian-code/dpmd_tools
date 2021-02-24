"""Read various data formats to deepmd.

Can be easily extended by writing new parser functions like e.g.
`read_vasp_out` and by altering get paths function.
"""

import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["KMP_WARNINGS"] = "FALSE"
from pathlib import Path
from typing import Any, List
from warnings import warn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from dpmd_tools.data import MultiSystemsVar
from dpmd_tools.frame_filter import ApplyConstraint
from dpmd_tools.readers import (read_dpmd_raw, read_vasp_dir, read_vasp_out,
                                read_xtalopt_dir)

WORK_DIR = Path.cwd()
PARSER_CHOICES = ("xtalopt", "vasp_dirs", "vasp_files", "dpmd_raw")


def input_parser():
    p = argparse.ArgumentParser(
        description="Load various data formats to deepmd. Loaded data will be output "
        "to deepmd_data/all - (every read structure) and deepmd_data/for_train - (only "
        "selected structures) dirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    required_dev = "-g " in sys.argv or "--graphs" in sys.argv

    p.add_argument("-p", "--parser", default=None, required=True, type=str,
                   choices=PARSER_CHOICES, help="input parser you wish to use")
    p.add_argument("-g", "--graphs", default=[], type=str, nargs="*",
                   help="input list of graphs you wish to use for checking if "
                   "datapoint is covered by current model. If present must "
                   "have at least two distinct graphs")
    p.add_argument("-e", "--every", default=None, type=int, help="take every "
                   "n-th frame")
    p.add_argument("-v", "--volume", default=None, type=float, nargs=2,
                   help="constrain structures volume. Input as 10.0 31. "
                   "In [A^3]")
    p.add_argument("-n", "--energy", default=None, type=float, nargs=2,
                   help="constrain structures energy. Input as -5 -2. In [eV]")
    p.add_argument("-a", "--per-atom", default=False, action="store_true",
                   help="set if energy, energy-dev and volume constraints are "
                   "computed per atom or for the whole structure")
    p.add_argument("-gp", "--get-paths", default=None, type=str, help="if not "
                   "specified default function will be used. Otherwise you can input "
                   "python code as string that outputs list of 'Path' objects. The "
                   "Path object is already imported for you. Example: "
                   "-g '[Path.cwd() / \"OUTCAR\"]'")
    p.add_argument("-de", "--dev-energy", default=False, required=required_dev,
                   type=float, nargs=2, help="specify energy deviations lower "
                   "and upper bound for selection")
    p.add_argument("-df", "--dev-force", default=False, required=required_dev,
                   type=float, nargs=2, help="specify force deviations lower "
                   "and upper bound for selection")
    p.add_argument("-m", "--mode", default="new", choices=("new", "append"),
                   type=str, help="choose data export mode in append "
                   "structures will be appended to ones already chosen for "
                   "training in previous iteration. In append mode do not specify the "
                   "-gp/--get-paths arguments and start the script in deepmd_data dir, "
                   "in this mode only dpmd_raw data format is supported")
    p.add_argument("-f", "--fingerprint-use", default=False, action="store_true",
                   help="if max-select argument is used that this option specifies "
                   "that subsample will be selected based on fingerprints")
    p.add_argument("-ms", "--max-select", default=None, type=int, help="set max number "
                   "of structures that will be selected. If above conditions produce "
                   "more, subsample will be selected randomly, or based on "
                   "fingerprints if available")
    p.add_argument("-mf", "--min-frames", default=30, type=int, help="specify minimal "
                   "munber of frames a system must have. Smaller systems are deleted. "
                   "This is due to difficulties in partitioning and inefficiency of "
                   "DeepMD when working with such small data")

    args = vars(p.parse_args())
    if len(set(args["graphs"])) == 1:
        raise ValueError("If you want to filter structures based on current "
                         "graphs you must input at least two")

    return args


class Loglprint:

    def __init__(self, log_file: Path) -> None:
        log_file.parent.mkdir(exist_ok=True, parents=True)

        if log_file.exists():
            self.log_stream = log_file.open("a")
            fmt = "%d/%m/%Y %H:%M:%S"
            self.__call__("\n=========================================================")
            self.__call__(f"Appending logs: {datetime.now().strftime(fmt)}")
            self.__call__("=========================================================\n")
        else:
            self.log_stream = log_file.open("w")




    def __call__(self, msg: Any) -> Any:
        self.log_stream.write(f"{str(msg)}\n")
        print(msg)


def get_paths() -> List[Path]:

    # paths = [d for d in WORK_DIR.glob("*/") if d.is_dir()]

    
    paths = []
    for root, dirs, files in os.walk(WORK_DIR, topdown=True):

        for f in files:
            if "OUTCAR" in f:
                paths.append(Path(root) / f)


        delete = []
        for i, d in enumerate(dirs):
            if "analysis" in d or "traj" in d or "results" in d or "test" in d:
                delete.append(i)

            #if "50" not in d:
            #    delete.append(i)

        for i in sorted(set(delete), reverse=True):
            del dirs[i]
    
    # paths = [WORK_DIR]

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


def main():

    args = input_parser()

    if args["mode"] == "new":
        DPMD_DATA = WORK_DIR / "deepmd_data"
        DPMD_DATA_ALL = DPMD_DATA / "all"
        DPMD_DATA_TRAIN = DPMD_DATA / "for_train"
    elif args["mode"] == "append":
        # adjust file locations
        DPMD_DATA = WORK_DIR
        DPMD_DATA_ALL = DPMD_DATA / "all"
        DPMD_DATA_TRAIN = DPMD_DATA / "for_train"

    if args["get_paths"]:
        _locals = {}
        exec(
            f"try:\n"
            f"    paths = {args['get_paths']}\n"
            f"except Exception as e:\n"
            f"    exception = e",
            globals(),
            _locals
        )
        try:
            paths = _locals["paths"]
        except KeyError:
            raise RuntimeError(f"your get_paths script crashed: {_locals['exception']}")
    elif args["mode"] == "append":
        paths = [d for d in DPMD_DATA_ALL.glob("*") if (d / "box.raw").is_file()]
    else:
        paths = get_paths()

    lprint = Loglprint(DPMD_DATA / "README")

    print("Script was run with these arguments ------------------------------")
    for arg, value in args.items():
        lprint(f"{arg:20}: {value}")

    lprint("will read from these paths:")
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

    # here we will store the filtered structures that we want to use for
    # training
    chosen_sys = MultiSystemsVar()

    lprint(f"size before {sum([len(s) for s in multi_sys.values()])}")
    for k, system in multi_sys.items():
        constraints = ApplyConstraint(
            k, system, args["max_select"], args["fingerprint_use"], lprint
        )
        if args["energy"]:
            constraints.energy(
                bracket=args["energy"], per_atom=args["per_atom"]
            )
        if args["volume"]:
            constraints.volume(
                bracket=args["volume"], per_atom=args["per_atom"]
            )
        if args["graphs"]:
            constraints.dev_e(
                graphs=args["graphs"],
                bracket=args["dev_energy"],
                per_atom=args["per_atom"]
            )
            constraints.dev_f(graphs=args["graphs"], bracket=args["dev_force"])
        if args["every"]:
            constraints.every(n_th=args["every"])

        if args["mode"] == "append":
            lprint("will add structures from prevoius selections")
        chosen_sys.append(constraints.apply())
        lprint("*************************************************************")

    lprint(f"size after {sum([len(s) for s in chosen_sys.values()])}")

    if args["graphs"]:
        lprint("plotting std for energies and max atom forces")
        plot(chosen_sys)

    lprint(f"deleting systems with less than {args['min_frames']} structures ----")
    del_systems = []
    for name, system in chosen_sys.items():
        if len(system) < args["min_frames"]:
            del_systems.append(name)

    for s in del_systems:
        lprint(f"deleting {s}")
        chosen_sys.systems.pop(s, None)

    lprint("shuffling systems -----------------------------------------------")
    chosen_sys.shuffle()

    # create dirs
    DPMD_DATA_ALL.mkdir(exist_ok=True, parents=True)
    DPMD_DATA_TRAIN.mkdir(exist_ok=True, parents=True)

    lprint("saving data for training ----------------------------------------")
    chosen_sys.to_deepmd_raw(DPMD_DATA_TRAIN)
    chosen_sys.to_deepmd_npy(DPMD_DATA_TRAIN, set_size="auto")

    lprint("saving all data for further use ---------------------------------")
    multi_sys.to_deepmd_raw(DPMD_DATA_ALL)

    lprint(f"data output to {DPMD_DATA}")

if __name__ == "__main__":
    main()
