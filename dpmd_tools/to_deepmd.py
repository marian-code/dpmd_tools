"""Read various data formats to deepmd.

Can be easily extended by writing new parser functions like e.g.
`read_vasp_out` and by altering get paths function.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from colorama import Fore, init
from dpmd_tools.utils import init_yappi

from dpmd_tools.system import MaskedSystem, SelectedSystem, MultiSystemsVar
from dpmd_tools.frame_filter import ApplyConstraint
from dpmd_tools.readers import (
    read_dpmd_raw,
    read_vasp_dir,
    read_vasp_out,
    read_xtalopt_dir,
)
from dpmd_tools.utils import BlockPBS, Loglprint, get_graphs

WORK_DIR = Path.cwd()
PARSER_CHOICES = ("xtalopt", "vasp_dirs", "vasp_files", "dpmd_raw")

init(autoreset=True)


def input_parser():
    p = argparse.ArgumentParser(
        description="Load various data formats to deepmd. Loaded data will be output "
        "to deepmd_data/all - (every read structure) and deepmd_data/for_train - (only "
        "selected structures) dirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-p",
        "--parser",
        default=None,
        required=True,
        type=str,
        choices=PARSER_CHOICES,
        help="input parser you wish to use",
    )
    p.add_argument(
        "-g",
        "--graphs",
        default=[],
        type=str,
        nargs="*",
        help="input list of graphs you wish to use for checking if "
        "datapoint is covered by current model. If present must "
        "have at least two distinct graphs. You can use glob patterns "
        "relative to current path e.g. '../ge_all_s1[3-6].pb'. Files can "
        "also be located on remote e.g. "
        "'kohn@'/path/to/file/ge_all_s1[3-6].pb'",
    )
    p.add_argument(
        "-e", "--every", default=None, type=int, help="take every n-th frame"
    )
    p.add_argument(
        "-v",
        "--volume",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures volume. Input as 10.0 31. In [A^3]",
    )
    p.add_argument(
        "-n",
        "--energy",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures energy. Input as -5 -2. In [eV]",
    )
    p.add_argument(
        "-a",
        "--per-atom",
        default=False,
        action="store_true",
        help="set if energy, energy-dev and volume constraints are "
        "computed per atom or for the whole structure",
    )
    p.add_argument(
        "-gp",
        "--get-paths",
        default=None,
        type=str,
        help="if not "
        "specified default function will be used. Otherwise you can input "
        "python code as string that outputs list of 'Path' objects. The "
        "Path object is already imported for you. Example: "
        "-g '[Path.cwd() / \"OUTCAR\"]'",
    )
    p.add_argument(
        "-de",
        "--dev-energy",
        default=False,
        type=float,
        nargs=2,
        help="specify energy deviations lower and upper bound for selection",
    )
    p.add_argument(
        "-df",
        "--dev-force",
        default=False,
        type=float,
        nargs=2,
        help="specify force deviations lower and upper bound for selection",
    )
    p.add_argument(
        "-m",
        "--std-method",
        default=False,
        action="store_true",
        help="method to use in forces and energy error estimation. Default=False means "
        "that root mean squared prediction error will be used, this will output high "
        "error even if all models predictions aggre but have a constant shift from DFT "
        "data. If true than insted standard deviation in set of predictions by "
        "different models will be used, this will not account for any prediction "
        "biases.",
    )
    p.add_argument(
        "-m",
        "--mode",
        default="new",
        choices=("new", "append"),
        type=str,
        help="choose data export mode in append "
        "structures will be appended to ones already chosen for "
        "training in previous iteration. In append mode do not specify the "
        "-gp/--get-paths arguments and start the script in deepmd_data dir, "
        "in this mode only dpmd_raw data format is supported",
    )
    p.add_argument(
        "-f",
        "--fingerprint-use",
        default=False,
        action="store_true",
        help="if max-select argument is used that this option specifies "
        "that subsample will be selected based on fingerprints",
    )
    p.add_argument(
        "-ms",
        "--max-select",
        default=None,
        type=str,
        help="set max number "
        "of structures that will be selected. If above conditions produce "
        "more, subsample will be selected randomly, or based on "
        "fingerprints if available. Can be also input as a percent of all "
        "dataset e.g. 10%% from 5000 = 500 frams selected. The percent "
        "option computes the potrion from whole dataset length not only "
        "from unselected structures",
    )
    p.add_argument(
        "-mf",
        "--min-frames",
        default=30,
        type=int,
        help="specify minimal "
        "munber of frames a system must have. Smaller systems are deleted. "
        "This is due to difficulties in partitioning and inefficiency of "
        "DeepMD when working with such small data",
    )
    p.add_argument(
        "-nf",
        "--n-from-cluster",
        default=100,
        type=int,
        help="number of random samples to select from each cluster",
    )
    p.add_argument(
        "-cp",
        "--cache-predictions",
        default=False,
        action="store_true",
        help="if true than prediction for current graphs are stored in "
        "running directory so they do not have to be recomputed when "
        "you wish to run the scrip again",
    )
    p.add_argument(
        "--auto",
        default=False,
        action="store_true",
        help="automatically accept when prompted to save changes",
    )
    p.add_argument(
        "--dont-save",
        default=False,
        action="store_true",
        help="if this switch is enabled only run and dont save selection, usefull "
        "for situations when one wants to precompute predictions",
    )
    p.add_argument(
        "-b",
        "--block-pbs",
        default=False,
        action="store_true",
        help="put an empty job in PBS queue to stop others from trying to access GPU",
    )
    p.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="profile this run with yappi",
    )

    args = vars(p.parse_args())

    args["graphs"] = get_graphs(args["graphs"])

    if args["auto"] and args["dont_save"]:
        raise ValueError("cannot pass both 'auto' and 'dont-save' arguments at once")

    if args["parser"] == "lmp_traj_dev":
        if not args["dev_energy"] and not args["dev_force"]:
            raise ValueError(
                "Must specify alt least one of the dev-energy/dev-force conditions"
            )
        if not args["per_atom"]:
            raise ValueError("per atoms must be true in this mode")

    if args["max_select"] is not None:
        if not args["max_select"].replace("%", "").isdigit():
            raise TypeError(
                "--max-select argument was specified with wrong format, use number or %"
            )

    if len(set(args["graphs"])) == 1:
        raise ValueError(
            "If you want to filter structures based on current "
            "graphs you must input at least two"
        )

    if (args["dev_force"] or args["dev_energy"]) and not args["graphs"]:
        raise RuntimeError(
            "if --dev-force or --dev-energy is specified you must input also --graphs "
            "argument. If out input --graphs argument then 'get_graphs' function has "
            "not found any graph files based on your input."
        )

    return args


def get_paths() -> List[Path]:

    paths = []
    for root, dirs, files in os.walk(WORK_DIR, topdown=True):

        for f in files:
            if "OUTCAR" in f:
                paths.append(Path(root) / f)

        delete = []
        for i, d in enumerate(dirs):
            if "analysis" in d or "traj" in d or "results" in d or "test" in d:
                delete.append(i)

        for i in sorted(set(delete), reverse=True):
            del dirs[i]

    return paths


def make_df(multi_sys: MultiSystemsVar) -> Tuple[pd.DataFrame, bool, bool]:

    has_e_std = all(["energies_std" in s.data for s in multi_sys])
    has_f_std = all(["forces_std_max" in s.data for s in multi_sys])

    dataframes = []
    for system in multi_sys:
        n = system.get_natoms()
        df = pd.DataFrame(
            {
                "energies": system.data["energies"].flatten() / n,
                "volumes": np.linalg.det(system.data["cells"]) / n,
            }
        )

        if has_e_std:
            df["energies_std"] = system.data["energies_std"].flatten()
        if has_f_std:
            df["forces_std_max"] = system.data["forces_std_max"]
        dataframes.append(df)

    data = pd.concat(dataframes)

    return data, has_e_std, has_f_std


def plot(multi_sys: MultiSystemsVar, chosen_sys: MultiSystemsVar, *, histogram: bool):

    data, has_e_std, has_f_std = make_df(multi_sys)
    if histogram:
        data_chosen, has_e_std, has_f_std = make_df(chosen_sys)

    for what in ("energies_std", "forces_std_max"):
        if what == "energies_std" and not has_e_std:
            print(
                f" - {Fore.YELLOW}skipping plot energies std, "
                f"this was not selection criterion"
            )
            continue
        elif what == "forces_std_max" and not has_f_std:
            print(
                f" - {Fore.YELLOW}skipping plot forces std max, "
                f"this was not selection criterion"
            )
            continue
        if histogram:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=data[what], histnorm="probability", name="all data in dataset"
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=data_chosen[what],
                    histnorm="probability",
                    name="data chosen from dataset",
                )
            )
            fig.update_layout(
                title=f"Histogram of {what} for all structures, 0 are for those selected in previous iterations",
                xaxis_title=what,
            )
        else:
            fig = go.Figure(
                data=go.Scattergl(
                    x=data["energies"],
                    y=data["volumes"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=data[what],
                        colorbar=dict(title=what),
                        colorscale="thermal",
                        showscale=True,
                    ),
                )
            )
            fig.update_layout(
                title=f"E-V plot, zero {what} are for structures selected in previous iterations",
                xaxis_title="V [A^3]",
                yaxis_title="E [eV]",
                coloraxis_colorbar=dict(title=what,),
            )
        fig.write_html(
            f"{what}{'_hist' if histogram else ''}_it{multi_sys.iteration}.html",
            include_plotlyjs="cdn",
        )


def main():  # NOSONAR

    args = input_parser()

    if args["block_pbs"]:
        BlockPBS()

    if args["profile"]:
        print("Profiling script with yappi")
        init_yappi()

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
            _locals,
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

    lprint(f"{Fore.GREEN}Script was run with these arguments -------------------------")
    for arg, value in args.items():
        lprint(f" - {arg:20}: {value}")

    lprint(f"{Fore.GREEN}will read from these paths:")
    for p in paths:
        lprint(f" - {p}")

    if not args["graphs"]:
        warn(
            f"{Fore.YELLOW}It is strongly advised to use filtering based on currently "
            f"trained model",
            UserWarning,
        )

    multi_sys = MultiSystemsVar[MaskedSystem]()

    if args["parser"] == "xtalopt":
        multi_sys.collect_cf(paths, read_xtalopt_dir)
    elif args["parser"] == "vasp_dirs":
        multi_sys.collect_cf(paths, read_vasp_dir)
    elif args["parser"] == "vasp_files":
        multi_sys.collect_cf(paths, read_vasp_out)
    elif args["parser"] == "dpmd_raw":
        multi_sys.collect_cf(paths, read_dpmd_raw)
    else:
        raise NotImplementedError(
            f"{Fore.RED}parser for {Fore.RESET}{args['parser']}{Fore.RED} "
            f"is not implemented"
        )

    if multi_sys.get_nframes() == 0:
        lprint(f"{Fore.RED}aborting, no data was found")
        sys.exit()

    lprint(f"{Fore.GREEN}got these systems -------------------------------------------")
    for name, system in multi_sys.items():
        lprint(f" - {name:6} -> {len(system):4} structures")

    lprint(f"{Fore.GREEN}filtering data ----------------------------------------------")

    # here we will store the filtered structures that we want to use for
    # training
    chosen_sys = MultiSystemsVar[SelectedSystem]()

    lprint(
        f"{Fore.LIGHTBLUE_EX}Number of frames before filtering{Fore.RESET} "
        f"{multi_sys.get_nframes()}"
    )
    for k, system in multi_sys.items():
        lprint(f"{Fore.LIGHTGREEN_EX}filtering system {k} ****************************")
        constraints = ApplyConstraint(
            system,
            args["fingerprint_use"],
            lprint,
            append=args["mode"] == "append",
            cache_predictions=args["cache_predictions"],
        )
        if args["energy"]:
            constraints.energy(bracket=args["energy"], per_atom=args["per_atom"])
        if args["volume"]:
            constraints.volume(bracket=args["volume"], per_atom=args["per_atom"])
        if args["graphs"] and args["dev_energy"]:
            constraints.dev_e(
                graphs=args["graphs"],
                bracket=args["dev_energy"],
                per_atom=args["per_atom"],
                std_method=args["std_method"],
            )
        if args["graphs"] and args["dev_force"]:
            constraints.dev_f(
                graphs=args["graphs"],
                bracket=args["dev_force"],
                std_method=args["std_method"],
            )
        if args["every"]:
            constraints.every(n_th=args["every"])
        if args["max_select"]:
            constraints.max_select(max_n=args["max_select"],)

        chosen_sys.append(constraints.apply())

    lprint(
        f"{Fore.LIGHTBLUE_EX}Number of frames after filtering {Fore.RESET}"
        f"{chosen_sys.get_nframes()},{Fore.LIGHTBLUE_EX} this includes previous "
        f"iterations selections"
    )

    if args["graphs"]:
        lprint(f"{Fore.GREEN}plotting std for energies and max atom forces")
        plot(multi_sys, chosen_sys, histogram=False)
        plot(multi_sys, chosen_sys, histogram=True)

    lprint(
        f"{Fore.GREEN}deleting systems with less than {Fore.RESET}{args['min_frames']} "
        f"{Fore.GREEN}structures ---------------"
    )
    del_systems = []
    for name, system in chosen_sys.items():
        if len(system) < args["min_frames"]:
            del_systems.append(name)

    for s in del_systems:
        lprint(f"deleting {s}")
        chosen_sys.systems.pop(s, None)

    if args["dont_save"]:
        lprint(f"{Fore.YELLOW}You choose not to save changes, exiting ... ")
        sys.exit()

    # if result is satisfactory continue, else abort
    if not args["auto"]:
        if input("Continue and write data to disk? [ENTER]") != "":  # NOSONAR
            lprint("selection run abborted, changes to dataset were not written")
            sys.exit()

    lprint(f"{Fore.GREEN}shuffling systems -------------------------------------------")
    chosen_sys.shuffle()

    # create dirs
    DPMD_DATA_ALL.mkdir(exist_ok=True, parents=True)
    DPMD_DATA_TRAIN.mkdir(exist_ok=True, parents=True)

    lprint(f"{Fore.GREEN}saving data for training ------------------------------------")
    chosen_sys.to_deepmd_npy(DPMD_DATA_TRAIN, set_size="auto")

    lprint(f"{Fore.GREEN}saving all data for further use -----------------------------")
    multi_sys.to_deepmd_raw(DPMD_DATA_ALL, True if args["mode"] == "append" else False)

    lprint(f"data output to {DPMD_DATA}")
    lprint.write()


if __name__ == "__main__":
    main()
