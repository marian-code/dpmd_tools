"""Compare E-V curves from ab-initio and deepMD

Can also plot MTD runs structures and dataset coverage.

You need to prepare directory structure for this to work:
|-phase1
 |-vasp
  |-vol_10
  |-vol_15
  |-vol_...
|-phase2
 |-vasp
  |-vol_12
  |-vol_xy

Each of vol_xy directories must contain vasp computed structure vith specified
volume.
"""

import re
import shutil
import subprocess
import sys
from operator import itemgetter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ase import Atoms, io, units
from ase.eos import EquationOfState
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from colorama import Fore, init
from dpmd_tools.utils import get_remote_files
from tqdm import tqdm
import pandas as pd
from ase.spacegroup import get_spacegroup, Spacegroup

sys.path.append("/home/rynik/OneDrive/dizertacka/code/rw")

try:
    from lammps import read_lammps_out
    from vasp import read_vasp_out
    from plotly_theme_setter import *
except ImportError:
    print("cannot use compare graph script, custom modules are missing")

init(autoreset=True)

COLORS = (
    "black",
    "blue",
    "purple",
    "green",
    "red",
    "cyan",
    "goldenrod",
    "gray",
    "lightpink",
    "firebrick",
    "limegreen",
    "navy",
    "pink",
)
WORK_DIR = Path.cwd()


def mod_lmp_in(graph: Path, lmp_in: str = "in.lammps"):

    print(f"using graph: {graph}")

    lmp_in = (WORK_DIR / "in.lammps").read_text()
    lmp_in = re.sub(r"\.\./\.\./\.\./\S*", f"../../../{graph.name}", lmp_in)
    (WORK_DIR / "in.lammps").write_text(lmp_in)

    print("lammps input modified")


def vasp_recompute(atom_style: str = "atomic", lmp_in: str = "in.lammps"):

    vasp: List[Path]
    vasp = [
        d
        for d in WORK_DIR.rglob("*/")
        if d.is_dir()
        and d.parent.name == "vasp"
        and not (d.parent.parent / ".noEV").is_file()
    ]

    lmp_binary = shutil.which("lmp")
    print(f"using lmp binary: {lmp_binary}")

    vasp_job = tqdm(vasp, ncols=100, total=len(vasp))

    for v in vasp_job:

        a: Atoms = io.read(v / "CONTCAR")

        l = v.parent.parent / "lammps" / v.name
        l.mkdir(exist_ok=True, parents=True)

        vasp_job.set_description(f"Computing {l.parent.parent.name}/{l.name}")

        write_lammps_data(str(l / "data.in"), a, atom_style=atom_style, force_skew=True)

        shutil.copy2(WORK_DIR / lmp_in, l)

        out = subprocess.run(
            [lmp_binary, "-in", lmp_in, "-nocite"],
            cwd=l,
            capture_output=True,
            encoding="utf-8",
        )

        try:
            out.check_returncode()
        except subprocess.CalledProcessError as e:
            vasp_job.write(str(e))
            vasp_job.write(out.stdout)


def collect_data_dirs(base_dir: Path, reference: str):

    collect_dirs = [d for d in base_dir.glob("*/") if d.is_dir()]
    collect_dirs = [d for d in collect_dirs if not (d / ".noEV").is_file()]

    # move Cubic diamond to the begining
    index = 0
    for i, c in enumerate(collect_dirs):
        if c.name == reference:
            index = i

    temp = collect_dirs[index]
    del collect_dirs[index]
    collect_dirs.insert(0, temp)

    return collect_dirs


def collect_lmp(collect_dirs: List[Path], lammpses: Tuple[str, ...]):

    iter_dirs = tqdm(collect_dirs, ncols=100, total=len(collect_dirs))

    for cd in iter_dirs:

        iter_dirs.set_description(f"get lmp data: {cd.parent.parent.name}/{cd.name}")

        for wd in [cd / l for l in lammpses]:

            dirs = [d for d in wd.glob("*/") if d.is_dir()]

            data = []

            for d in dirs:
                N = len(read(d / "data.out", format="lammps-data", style="atomic"))
                en, vol, stress = read_lammps_out(d / "log.lammps")[:3]
                stress *= units.bar / units.GPa
                data.append([vol / N, (en / N), stress])

            data = np.array(data)

            np.savetxt(
                wd / "vol_stress.txt",
                data[data[:, 0].argsort()],
                header="# Volume Energy stress",
                fmt="%.6f",
            )


def parse_vasp(path: Path) -> Tuple[float, float, float]:

    try:
        atoms = read(path)
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        hydrostatic_stress = -np.mean((atoms.get_stress()[:3] / units.GPa))
    except ValueError as e:
        print(path.parent, e)
        atoms = read(path.parent / "CONTCAR")
        energy, volume, hydrostatic_stress = read_vasp_out(path)[:3]
        hydrostatic_stress /= 10

    return energy, volume, hydrostatic_stress, get_spacegroup(atoms, symprec=0.01)


def collect_vasp(collect_dirs: List[Path]):

    iter_dirs = tqdm(collect_dirs, ncols=100, total=len(collect_dirs))

    for cd in iter_dirs:

        iter_dirs.set_description(f"get vasp data: {cd.parent.parent.name}/{cd.name}")

        wd = cd / "vasp"

        dirs = [d for d in wd.glob("*/") if d.is_dir()]

        data = []

        for d in dirs:
            N = len(read(d / "POSCAR"))
            try:
                en, vol, stress, spg = parse_vasp(d / "OUTCAR")
            except Exception as e:
                print(d, e)
                raise e from None
            data.append([vol / N, en / N, stress, spg])

        data = np.array(data)

        np.savetxt(
            wd / "vol_stress.txt",
            data[data[:, 0].argsort()],
            header="Volume Energy stress spg",
            fmt="%.6f",
        )


def plot_mpl(
    collect_dirs: List[Path],
    eos: str,
    lammpses: Tuple[str, ...],
    labels: Tuple[str, ...],
):

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):

        vasp_data = np.loadtxt(wd / "vasp" / "vol_stress.txt", skiprows=1, unpack=True)

        vasp_state = EquationOfState(vasp_data[0], vasp_data[1], eos=eos)
        x, y = itemgetter(4, 5)(vasp_state.getplotdata())

        plt.scatter(vasp_data[0], vasp_data[1], s=25, c=c)
        plt.plot(
            x, y, label=f"{wd} - Ab initio (VASP)", color=c, linestyle="-", linewidth=2
        )

        for l, lab, ls in zip(lammpses, labels, ("", ":", "-.", "-")):
            lmps_data = np.loadtxt(wd / l / "vol_stress.txt", skiprows=1, unpack=True)

            lmps_state = EquationOfState(lmps_data[0], lmps_data[1], eos=eos)
            x, y = itemgetter(4, 5)(lmps_state.getplotdata())
            plt.scatter(lmps_data[0], lmps_data[1], s=25, c=c)
            plt.plot(x, y, label=f"{wd} - {lab}", color=c, linestyle=ls, linewidth=2)

    plt.xlabel("volume per atom")
    plt.ylabel("energy per atom")
    plt.title(
        "E-V diagram for NNP, GAP and SNAP potentials vs " "Ab Initio calculations"
    )

    plt.tight_layout()

    plt.legend()
    plt.show()


def plot_abinit_ev(collect_dirs: List[Path], eos: str) -> go.Figure:

    fig = go.Figure()

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):

        vasp_data = pd.read_table(
            wd / "vasp" / "vol_stress.txt",
            sep=r"\s+",
            names=["volume", "energy", "stress", "spg"],
            header=0,
            comment="#",
        )

        vasp_state = EquationOfState(
            vasp_data["volume"].to_numpy(), vasp_data["energy"].to_numpy(), eos=eos
        )
        x, y = itemgetter(4, 5)(vasp_state.getplotdata())

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                name=f"{wd.name} - Ab initio (VASP)",
                line=dict(color=c, width=3, dash="solid"),
                legendgroup=f"vasp_{c}",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=vasp_data["volume"],
                y=vasp_data["energy"],
                hovertext=[
                    f"{wd.name}<br>Ab initio (VASP)<br>p={p:.3f}<br>V={v:.3f}<br>"
                    f"E={e:.3f}<br>spg={Spacegroup(int(s)).symbol:s}"
                    for v, e, p, s in vasp_data.itertuples(index=False)
                ],
                hoverinfo="text",
                mode="markers",
                showlegend=False,
                line=dict(color=c, width=3),
                marker_size=25,
                legendgroup=f"vasp_{c}",
            )
        )

    return fig


def plot_abinit_hp(collect_dirs: List[Path]) -> go.Figure:

    fig = go.Figure()

    reference = None

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):

        df = pd.read_table(
            wd / "vasp" / "vol_stress.txt",
            sep=r"\s+",
            names=["volume", "energy", "stress", "spg"],
            header=0,
            comment="#",
        )
        df = df.assign(enthalpy=df["energy"] + df["stress"] * units.GPa * df["volume"])

        vasp_state = np.poly1d(np.polyfit(df["stress"], df["enthalpy"], 1))

        # make reference from the first data point
        if not reference:
            reference = vasp_state

        fig.add_trace(
            go.Scattergl(
                x=df["stress"],
                y=vasp_state(df["stress"]) - reference(df["stress"]),
                name=f"{wd.name} - Ab initio (VASP)",
                line=dict(color=c, width=3, dash="solid"),
                marker=dict(size=15),
                mode="markers+lines",
                legendgroup=f"vasp_{c}",
            )
        )
        """
        fig.add_trace(
            go.Scattergl(
                x=df["stress"],
                y=df["enthalpy"] - reference(df["stress"]),
                name=f"{wd.name} - Ab initio (VASP)",
                mode="markers",
                showlegend=False,
                line=dict(color=c, width=3),
                marker_size=25,
                legendgroup=f"vasp_{c}",
            )
        )
        """

    return fig, reference


def plot_predicted_hp(
    collect_dirs: List[Path],
    lammpses: Tuple[str, ...],
    labels: Tuple[str, ...],
    fig: go.Figure,
    reference: np.poly1d,
) -> go.Figure:

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):
        for l, lab, ls in zip(lammpses, labels, ("dash", "dot", "dashdot", "-")):
            df = pd.read_table(
                wd / l / "vol_stress.txt",
                sep=r"\s+",
                names=["volume", "energy", "stress"],
                header=0,
                comment="#",
            )

            df = df.assign(
                enthalpy=df["energy"] + df["stress"] * units.GPa * df["volume"]
            )

            lmps_state = np.poly1d(np.polyfit(df["stress"], df["enthalpy"], 1))

            fig.add_trace(
                go.Scattergl(
                    x=df["stress"],
                    y=lmps_state(df["stress"]) - reference(df["stress"]),
                    name=f"{wd.name} - {lab}",
                    line=dict(color=c, width=3, dash=ls),
                    marker=dict(size=13, color="white", line=dict(width=2, color=c)),
                    mode="markers+lines",
                    legendgroup=c,
                )
            )
            """
            fig.add_trace(
                go.Scattergl(
                    x=df["stress"],
                    y=df["enthalpy"] - reference(df["stress"]),
                    name=f"{wd.name} - {lab}",
                    mode="markers",
                    showlegend=False,
                    legendgroup=c,
                    marker_line_width=3,
                    marker_symbol="circle-open",
                    marker_size=25,
                    line=dict(color=c, width=3),
                )
            )
            """

    return fig


def plot_predicted_ev(
    collect_dirs: List[Path],
    eos: str,
    lammpses: Tuple[str, ...],
    labels: Tuple[str, ...],
    fig: go.Figure,
) -> go.Figure:

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):
        for l, lab, ls in zip(lammpses, labels, ("dash", "dot", "dashdot", "-")):
            lmps_data = pd.read_table(
                wd / l / "vol_stress.txt",
                sep=r"\s+",
                names=["volume", "energy", "stress"],
                header=0,
                comment="#",
            )

            lmps_state = EquationOfState(
                lmps_data["volume"].to_numpy(), lmps_data["energy"].to_numpy(), eos=eos
            )
            try:
                x, y = itemgetter(4, 5)(lmps_state.getplotdata())
            except RuntimeError:
                print(f"{Fore.RED}Could not fit equation of state for {wd.name}")
                continue
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    name=f"{wd.name} - {lab}",
                    line=dict(color=c, width=3, dash=ls),
                    legendgroup=c,
                )
            )
            fig.add_trace(
                go.Scattergl(
                    x=lmps_data["volume"],
                    y=lmps_data["energy"],
                    name=f"{wd.name} - {lab}",
                    mode="markers",
                    showlegend=False,
                    legendgroup=c,
                    marker_line_width=3,
                    marker_symbol="circle-open",
                    marker_size=25,
                    line=dict(color=c, width=3),
                )
            )

    return fig


def get_coverage(data_dir: Path, types: str = "type.raw"):

    dirs = [d for d in data_dir.rglob("*") if (d / types).is_file()]

    iter_dirs: Iterator[Path] = tqdm(dirs, ncols=100, total=len(list(dirs)))

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for d in iter_dirs:
        iter_dirs.set_description(f"get coverage: {d.name}")
        n_atoms = len((d / "type.raw").read_text().splitlines())
        sets = sorted(d.glob("set.*"), key=lambda x: x.name.split(".")[1])
        energies = np.concatenate([np.load(s / "energy.npy") for s in sets]) / n_atoms
        cells = np.vstack([np.load(s / "box.npy") for s in sets]).reshape((-1, 3, 3))
        volumes = np.array([np.abs(np.linalg.det(c)) for c in cells]) / n_atoms

        if len(d.relative_to(data_dir).parts) == 1:
            # data/md_cd/...raw
            name = d.name.replace("data_", "")
        else:
            # data/md_cd/Ge136/...raw
            name = d.parent.name.replace("data_", "")
        if name in data:
            data[name]["energy"] = np.concatenate((data[name]["energy"], energies))
            data[name]["volume"] = np.concatenate((data[name]["volume"], volumes))
        else:
            data[name] = {"energy": energies, "volume": volumes}

    return dict(sorted(data.items()))


def get_mtd_runs(paths: List[str]):

    paths = get_remote_files(paths, remove_after=True, same_names=True)

    data = {}
    for p in paths:
        if len(p.suffixes) == 2:
            name = p.suffixes[0][1:]
        else:
            name = p.parent.name
        data[name] = np.load(p)

    return data


def run(args, graph: Optional[Path]):
    RECOMPUTE = args["recompute"]
    EQ = args["equation"]
    TRAIN_DIR = args["train_dir"]
    MTD_RUNS = args["mtds"]
    ABINIT_DIR = WORK_DIR if not args["abinit_dir"] else Path(args["abinit_dir"])

    collect_dirs = collect_data_dirs(ABINIT_DIR, reference=args["reference_structure"])
    assert len(collect_dirs) <= len(COLORS), "There is not enough colors to plot"
    collect_vasp(collect_dirs)
    fig_ev = plot_abinit_ev(collect_dirs, EQ)
    fig_hp, reference = plot_abinit_hp(collect_dirs)

    if graph:
        mod_lmp_in(graph)

        # calculate nnp
        if RECOMPUTE:
            vasp_recompute()

        lammpses = ("lammps",)  # , "gc_lammps")
        labels = ("DPMD",)

        collect_lmp(collect_dirs, lammpses)

        fig_ev = plot_predicted_ev(collect_dirs, EQ, lammpses, labels, fig_ev)
        fig_ev.update_layout(
            title="E-V diagram for DeepMd potential vs Ab Initio calculations"
        )
        fig_hp = plot_predicted_hp(collect_dirs, lammpses, labels, fig_hp, reference)
        fig_hp.update_layout(
            title="H(p) plot for DeepMd potential vs Ab Initio calculations"
        )
        # plot_mpl(collect_dirs)
    else:
        graph = Path.cwd() / "graph"
        fig_ev.update_layout(title="dataset E-V coverage")

    fig_ev.update_layout(
        xaxis_title=f"V [{ANGSTROM}{POW3} / atom]",
        yaxis_title="E [eV / atom]",
        template="minimovka",
    )
    fig_hp.update_layout(
        xaxis_title=f"p [GPa]",
        yaxis_title=f"{DELTA}H [eV / atom], {collect_dirs[0].name} is reference",
        template="minimovka",
    )

    if MTD_RUNS:
        mtd_data = get_mtd_runs(MTD_RUNS)

        for name, mdata in mtd_data.items():
            fig_ev.add_trace(
                go.Scattergl(
                    x=mdata["volume"], y=mdata["energy"], mode="markers", name=name
                )
            )

    if TRAIN_DIR:
        for t in TRAIN_DIR:
            coverage_data = get_coverage(Path(t))

            for name, cdata in coverage_data.items():
                fig_ev.add_trace(
                    go.Scattergl(
                        x=cdata["volume"], y=cdata["energy"], mode="markers", name=name
                    )
                )

    fig_ev.write_html(f"E-V({graph.stem}).html", include_plotlyjs="cdn")
    fig_hp.write_html(f"H-p({graph.stem}).html", include_plotlyjs="cdn")


def compare_ev(args: dict):

    if args["graph"]:
        graphs = get_remote_files(args["graph"])
        for graph in graphs:
            print(f"analyzing for graph {graph}")
            print("----------------------------------------------------------")
            run(args, graph)
    else:
        run(args, None)
