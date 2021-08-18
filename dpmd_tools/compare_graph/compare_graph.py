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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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
from ssh_utilities import Connection, path_wildcard_expand

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
    "red",
    "green",
    "purple",
    "cyan",
    "goldenrod",
    "gray",
    "lightpink",
    "firebrick",
    "limegreen",
    "navy",
    "indigo",
    "khaki",
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


def parse_vasp(path: Path) -> Tuple[float, float, float, float]:

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

    return (
        energy,
        volume,
        hydrostatic_stress,
        get_spacegroup(atoms, symprec=0.01).no,
        len(atoms),
    )


def parse_qe(path: Path) -> Tuple[float, float, float, float]:

    atoms = read(path)
    energy = atoms.get_potential_energy()
    volume = atoms.get_volume()

    infile = (path.parent / "relax.in").read_text()
    hydrostatic_stress = float(re.findall(r"\s+press\s+=\s+(\S+)", infile)[0]) / 10

    return (
        energy,
        volume,
        hydrostatic_stress,
        get_spacegroup(atoms, symprec=0.01).no,
        len(atoms),
    )


def collect_vasp(collect_dirs: List[Path]):

    iter_dirs = tqdm(collect_dirs, ncols=100, total=len(collect_dirs))

    for cd in iter_dirs:

        iter_dirs.set_description(f"get vasp data: {cd.parent.parent.name}/{cd.name}")

        wd = cd / "vasp"

        dirs = [d for d in wd.glob("*/") if d.is_dir()]

        data = []

        for d in dirs:
            try:
                en, vol, stress, spg, N = parse_vasp(d / "OUTCAR")
            except FileNotFoundError:
                en, vol, stress, spg, N = parse_qe(d / "relax.out")
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


def plot_abinit_ev(
    collect_dirs: List[Path], eos: str, shift_ref: bool
) -> Tuple[go.Figure, Dict[str, float]]:

    fig = go.Figure()
    reference = None

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
        try:
            v0, e0, _ = vasp_state.fit()
        except RuntimeError:
            print(f"Could not fit EOS for {wd}")
            continue

        if shift_ref:
            if reference is None:
                reference = {"v0": v0, "e0": e0}
        else:
            reference = {"v0": 1, "e0": 0}
        x, y = itemgetter(4, 5)(vasp_state.getplotdata())

        fig.add_trace(
            go.Scattergl(  # type: ignore
                x=x / reference["v0"],
                y=y - reference["e0"],
                name=f"{wd.name} - Ab initio (VASP)",
                line=dict(color=c, width=3, dash="solid"),
                legendgroup=f"vasp_{c}",
            )
        )
        fig.add_trace(
            go.Scattergl(  # type: ignore
                x=vasp_data["volume"] / reference["v0"],
                y=vasp_data["energy"] - reference["e0"],
                hovertext=[
                    f"{wd.name}<br>Ab initio (VASP)<br>p={p:.3f}<br>V={v:.3f}<br>"
                    f"E={e:.3f}<br>spg={Spacegroup(int(s)).symbol:s}"
                    for v, e, p, s in vasp_data.itertuples(index=False)
                ],
                hoverinfo="text",
                mode="markers",
                showlegend=False,
                line=dict(color=c, width=3),
                marker_size=15,
                legendgroup=f"vasp_{c}",
            )
        )

    return fig, reference


class BM:
    """Birch–Murnaghan equation of state.

    References
    ----------
    https://en.wikipedia.org/wiki/Birch%E2%80%93Murnaghan_equation_of_state
    """

    def __init__(self, b0: float, bp: float, v0: float) -> None:
        self.b0 = b0
        self.bp = bp
        self.v0 = v0

    def __call__(self, volumes: np.ndarray) -> np.ndarray:

        dV = self.v0 / volumes
        press = (
            (3 * self.b0)
            / 2
            * (np.power(dV, 7 / 3) - np.power(dV, 5 / 3))
            * (1 + (3 / 4) * (self.bp - 4) * (np.power(dV, 2 / 3) - 1))
        )
        return press


def plot_abinit_hp(collect_dirs: List[Path], *, eos: str, fit: bool) -> go.Figure:

    fig = go.Figure()
    figm, ax = plt.subplots()

    reference = None

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):

        df = pd.read_table(
            wd / "vasp" / "vol_stress.txt",
            sep=r"\s+",
            names=["volume", "energy", "stress", "spg"],
            header=0,
            comment="#",
        )

        if fit:
            vasp_state = EquationOfState(
                df["volume"].to_numpy(), df["energy"].to_numpy(), eos=eos
            )
            try:
                vasp_state.fit()
            except RuntimeError:
                print(f"Could not fit EOS for {wd}")
                continue
            _, B0, BP, V0 = vasp_state.eos_parameters
            print(wd.name, "------------------------")
            print(f"B_0: {B0 / units.GPa:.2f}, B'_0: {BP:.2f}")
            df["stress"] = BM(B0, BP, V0)(df["volume"].to_numpy()) / units.GPa

        df = df.assign(enthalpy=df["energy"] + df["stress"] * units.GPa * df["volume"])

        vasp_state = np.poly1d(np.polyfit(df["stress"], df["enthalpy"], 1))

        # make reference from the first data point
        if not reference:
            reference = vasp_state
            # ! testing
            # def ref(x):
            #    return np.zeros(x.shape)
            # reference = ref

        fig.add_trace(
            go.Scattergl(  # type: ignore
                x=df["stress"],
                y=vasp_state(df["stress"]) - reference(df["stress"]),
                name=f"{wd.name} - Ab initio (VASP)",
                line=dict(color=c, width=3, dash="solid"),
                marker=dict(size=15),
                mode="markers+lines",
                legendgroup=f"vasp_{c}",
            )
        )
        fig.add_trace(
            go.Scattergl(  # type: ignore
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

        fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=10))

        # ax.scatter(df["stress"], df["enthalpy"] - reference(df["stress"]), s=15, c=c)
        ax.plot(
            df["stress"],
            vasp_state(df["stress"]) - reference(df["stress"]),
            label=f"{wd.name} - Ab initio (VASP)",
            color=c,
            linestyle="solid",
            linewidth=1,
        )

    return fig, reference, figm, ax


def plot_predicted_hp(
    collect_dirs: List[Path],
    lammpses: Tuple[str, ...],
    labels: Tuple[str, ...],
    fig: go.Figure,
    reference: np.poly1d,
    *,
    eos: str,
    fit: bool,
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

            if fit:
                vasp_state = EquationOfState(
                    df["volume"].to_numpy(), df["energy"].to_numpy(), eos=eos
                )
                vasp_state.fit()
                _, B0, BP, V0 = vasp_state.eos_parameters
                df["stress"] = BM(B0, BP, V0)(df["volume"].to_numpy()) / units.GPa

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
    reference: Dict[str, float],
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
                go.Scattergl(  # type: ignore
                    x=x / reference["v0"],
                    y=y - reference["e0"],
                    name=f"{wd.name} - {lab}",
                    line=dict(color=c, width=3, dash=ls),
                    legendgroup=c,
                )
            )
            fig.add_trace(
                go.Scattergl(  # type: ignore
                    x=lmps_data["volume"] / reference["v0"],
                    y=lmps_data["energy"] - reference["e0"],
                    name=f"{wd.name} - {lab}",
                    mode="markers",
                    showlegend=False,
                    legendgroup=c,
                    marker_line_width=3,
                    marker_symbol="circle-open",
                    marker_size=15,
                    line=dict(color=c, width=3),
                )
            )

    return fig


def get_coverage(data_dir: Path, types: str = "type.raw"):

    dirs = [d for d in data_dir.glob("**") if (d / types).is_file()]

    print(data_dir, type(data_dir))
    print(dirs)

    iter_dirs: Iterator[Path] = tqdm(dirs, ncols=100, total=len(list(dirs)))

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for d in iter_dirs:
        if "minima" in str(d):
            continue
        iter_dirs.set_description(f"get coverage: {d.name}")
        print("readlines")
        n_atoms = len((d / "type.raw").read_text().splitlines())
        sets = sorted(d.glob("set.*"), key=lambda x: x.name.split(".")[1])
        print("np load")
        energies = (
            np.concatenate([np.load((s / "energy.npy").open("rb")) for s in sets])
            / n_atoms
        )
        cells = np.vstack([np.load((s / "box.npy").open("rb")) for s in sets]).reshape(
            (-1, 3, 3)
        )
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

    if RECOMPUTE and not graph:
        raise ValueError("Must input at least one graph if you want to recompute")

    collect_dirs = collect_data_dirs(ABINIT_DIR, reference=args["reference_structure"])
    assert len(collect_dirs) <= len(COLORS), (
        f"There is not enough colors to plot, "
        f"You are missing {len(collect_dirs) - len(COLORS)} more color(s)"
    )
    collect_vasp(collect_dirs)
    fig_ev, reference_ev = plot_abinit_ev(collect_dirs, EQ, shift_ref=args["shift_ev"])
    fig_hp, reference_hp, figm_hp, ax_hp = plot_abinit_hp(
        collect_dirs, eos=EQ, fit=False
    )

    if graph:
        mod_lmp_in(graph)

        # calculate nnp
        if RECOMPUTE:
            vasp_recompute()

        lammpses = ("lammps",)  # , "gc_lammps")
        labels = ("DPMD",)

        collect_lmp(collect_dirs, lammpses)

        fig_ev = plot_predicted_ev(
            collect_dirs, EQ, lammpses, labels, fig_ev, reference_ev
        )
        fig_ev.update_layout(
            title="E-V diagram for DeepMd potential vs Ab Initio calculations"
        )
        fig_hp = plot_predicted_hp(
            collect_dirs, lammpses, labels, fig_hp, reference_hp, eos=EQ, fit=True
        )
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
    ax_hp.set(
        xlabel=f"p [GPa]",
        ylabel=f"{DELTA}H [eV / atom], {collect_dirs[0].name} is reference",
    )
    ax_hp.grid(linewidth=0.5)
    ax_hp.yaxis.set_minor_locator(AutoMinorLocator())
    ax_hp.xaxis.set_minor_locator(AutoMinorLocator())
    ax_hp.set_xlim(-10, 40)
    ax_hp.set_ylim(-1.5, 1)
    figm_hp.legend(loc="upper left", bbox_to_anchor=(0.25, 0.35), borderaxespad=0.0)

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
            if "@" in t:
                server, t = t.split("@")
                local = False
            else:
                server = "LOCAL"
                local = True

            print(server, local)
            with Connection(server, quiet=True, local=local) as c:
                print(c)
                path = c.pathlib.Path(t)
                coverage_data = get_coverage(path)

            for name, cdata in coverage_data.items():
                fig_ev.add_trace(
                    go.Scattergl(
                        x=cdata["volume"], y=cdata["energy"], mode="markers", name=name
                    )
                )

    filename = f"({graph.stem}-{args['reference_structure']}).html"
    print(f"writing: {filename}")

    fig_ev.write_html(f"E-V{filename}", include_plotlyjs="cdn")
    fig_hp.write_html(f"H-p{filename}", include_plotlyjs="cdn")
    figm_hp.savefig(
        f"H-p{filename}".replace("html", "png"), dpi=500, bbox_inches="tight"
    )


def compare_ev(args: dict):

    if args["graph"]:
        graphs = get_remote_files(args["graph"])
        for graph in graphs:
            print(f"analyzing for graph {graph}")
            print("----------------------------------------------------------")
            run(args, graph)
    else:
        run(args, None)
