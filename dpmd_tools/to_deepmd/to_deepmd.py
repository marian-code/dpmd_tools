"""Read various data formats to deepmd.

Can be easily extended by writing new parser functions like e.g.
`read_vasp_file` and by altering get paths function.
"""

import os
import sys
from collections import deque
from pathlib import Path
from time import sleep
from typing import List, Tuple
from warnings import warn

import dpmd_tools.readers.to_dpdata as readers
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from colorama import Fore, init
from dpmd_tools.system import MaskedSystem, MultiSystemsVar, SelectedSystem
from dpmd_tools.utils import BlockPBS, Loglprint, get_remote_files, init_yappi

from .frame_filter import ApplyConstraint

WORK_DIR = Path.cwd()
PARSER_CHOICES = [r.replace("read_", "") for r in readers.__all__]
COLLECTOR_CHOICES = [
    v.replace("collect_", "") for v in vars(MultiSystemsVar) if "collect_" in v
]

init(autoreset=True)


def postporcess_args(args: dict):

    args["graphs"] = get_remote_files(args["graphs"], remove_after=True)

    if args["parser"] == "lmp_traj_dev":
        if not args["dev_energy"] and not args["dev_force"]:
            raise ValueError(
                "Must specify alt least one of the dev-energy/dev-force conditions"
            )
        if not args["per_atom"]:
            raise ValueError("per atoms must be true in this mode")

    if args["max_select"] is not None:
        if not args["max_select"].replace("%", "").isdigit():  # NOSONAR
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
            "argument. If out input --graphs argument then 'get_remote_files' function has "
            "not found any graph files based on your input."
        )

    if len(args["take_slice"]) == 2 or args["take_slice"][0] == None:
        args["take_slice"] = slice(*args["take_slice"])
    else:
        args["take_slice"] = args["take_slice"][0]

    return args


def wait(path: Path):

    loader = deque(["-", "/", "|", "\\"])

    while True:
        if path.exists():
            print(f"Path {path} is present starting computation")
            sleep(5)
            return
        else:
            print(
                f"{Fore.GREEN}Waiting for {Fore.RESET}{path}{Fore.GREEN} to become "
                f"available {Fore.RESET}{loader[0]}",
                end="\r"
            )
            loader.rotate(1)
            sleep(0.15)


def get_paths() -> List[Path]:

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

        for i in sorted(set(delete), reverse=True):
            del dirs[i]
    """
    paths = [Path(str(i)) for i in range(6000, 6050)]

    return paths


def make_df(multi_sys: MultiSystemsVar) -> Tuple[pd.DataFrame, bool, bool]:

    has_dev_e = all([s.has_dev_e for s in multi_sys])
    has_dev_f = all([s.has_dev_f for s in multi_sys])

    dataframes = []
    for system in multi_sys:
        n = system.get_natoms()
        df = pd.DataFrame(
            {
                "energies": system.data["energies"].flatten() / n,
                "volumes": np.linalg.det(system.data["cells"]) / n,
            }
        )

        if has_dev_e:
            df["energies_std"] = system.data["energies_std"].flatten()
        if has_dev_f:
            df["forces_std"] = system.data["forces_std"]
        dataframes.append(df)

    data = pd.concat(dataframes)

    return data, has_dev_e, has_dev_f


def plot(multi_sys: MultiSystemsVar, chosen_sys: MultiSystemsVar, *, histogram: bool):

    data, has_dev_e, has_dev_f = make_df(multi_sys)
    if histogram:
        data_chosen, has_dev_e, has_dev_f = make_df(chosen_sys)

    for what in ("energies_std", "forces_std"):
        if what == "energies_std" and not has_dev_e:
            print(
                f" - {Fore.YELLOW}skipping plot energies std, "
                f"this was not selection criterion"
            )
            continue
        elif what == "forces_std" and not has_dev_f:
            print(
                f" - {Fore.YELLOW}skipping plot forces std max, "
                f"this was not selection criterion"
            )
            continue
        if histogram:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=data[what][data[what] > 0], histnorm="probability", name="all data in dataset"
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=data_chosen[what][data_chosen[what] > 0],
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
                coloraxis_colorbar=dict(
                    title=what,
                ),
            )

        name = Path(f"{what}{'_hist' if histogram else ''}_it{multi_sys.iteration}.html")
        index = 1
        while True:
            if name.is_file():
                name = name.with_suffix(f".{index}.html")
                index += 1
            else:
                break
        fig.write_html(
            str(name),
            include_plotlyjs="cdn",
        )


def to_deepmd(args: dict):  # NOSONAR

    if args["wait_for"]:
        wait(Path(args["wait_for"]))

    args = postporcess_args(args)

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
    elif args["mode"] == "merge":
        DPMD_DATA = WORK_DIR
        DPMD_DATA_ALL = None
        DPMD_DATA_TRAIN = args["merge_dir"]

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
            paths = list(_locals["paths"])
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

    if len(paths) > 20:
        for p in paths[:10]:
            lprint(f" - {p}")
        lprint(" - ...")
        for p in paths[-10:]:
            lprint(f" - {p}")
    else:
        for p in paths:
            lprint(f" - {p}")

    if not args["graphs"]:
        warn(
            f"{Fore.YELLOW}It is strongly advised to use filtering based on currently "
            f"trained model",
            UserWarning,
        )

    multi_sys = MultiSystemsVar[MaskedSystem]()

    collector = getattr(multi_sys, f"collect_{args['data_collector']}")
    try:
        reader = getattr(readers, f"read_{args['parser']}")
    except AttributeError:
        raise NotImplementedError(
            f"{Fore.RED}parser for {Fore.RESET}{args['parser']}{Fore.RED} "
            f"is not implemented"
        )
    else:
        slice_idx = args.pop("take_slice")
        collector(paths, reader, slice_idx, **args)

    if multi_sys.get_nframes() == 0:
        lprint(f"{Fore.RED}aborting, no data was found")
        sys.exit()

    lprint(f"{Fore.GREEN}got these systems -------------------------------------------")
    for name, system in multi_sys.items():
        lprint(f" - {name:6} -> {len(system):4} structures")

    if args["mode"] == "merge":
        lprint(f"{Fore.GREEN}merging data ----------------------------------------------")
    else:
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
        if args["pressure"]:
            constraints.pressure(bracket=args["pressure"])
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
            constraints.max_select(
                max_n=args["max_select"],
            )

        if args["mode"] == "merge":
            constraints.previous_iteration()

        # if dev_e or dev_f is used system class is mutated to other base with new
        # attributes that are needed for plotting
        if constraints.system_mutated:
            multi_sys[k] = constraints.system

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

    if args["save"] == "no":
        lprint(f"{Fore.YELLOW}You choose not to save changes, exiting ... ")
        sys.exit()
    # if result is satisfactory continue, else abort
    elif args["save"] == "input":
        if input("Continue and write data to disk? [ENTER]") != "":  # NOSONAR
            lprint("selection run abborted, changes to dataset were not written")
            sys.exit()

    lprint(f"{Fore.GREEN}shuffling systems -------------------------------------------")
    chosen_sys.shuffle()

    # create dirs
    if args["mode"] != "merge":
        DPMD_DATA_ALL.mkdir(exist_ok=True, parents=True)
    DPMD_DATA_TRAIN.mkdir(exist_ok=True, parents=True)

    lprint(f"{Fore.GREEN}saving data for training ------------------------------------")
    chosen_sys.to_deepmd_npy(DPMD_DATA_TRAIN, set_size="auto")

    if args["mode"] != "merge":
        lprint(f"{Fore.GREEN}saving all data for further use -----------------------------")
        multi_sys.to_deepmd_raw(DPMD_DATA_ALL, True if args["mode"] == "append" else False)

    if args["mode"] == "merge":
        lprint(f"merged data to {DPMD_DATA_TRAIN}")
    else:
        lprint(f"data output to {DPMD_DATA}")
        lprint.write()
