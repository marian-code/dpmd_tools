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
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms, io, units
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.spacegroup import get_spacegroup
from colorama import init
from dpmd_tools.utils import get_remote_files
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm

from .abinit_plot import plot_abinit_ev, plot_abinit_hp
from .predicted_plot import plot_predicted_ev, plot_predicted_hp
from . import COLORS
from .coverage_plot import plot_coverage_data
from .mtd_traj_plot import plot_mtd_data

sys.path.append("/home/rynik/OneDrive/dizertacka/code/rw")

try:
    from lammps import read_lammps_out
    from vasp import read_vasp_out
    from plotly_theme_setter import *
except ImportError:
    print("cannot use compare graph script, custom modules are missing")
    ANGSTROM = "A"
    POW3 = "3"
    DELTA = "d"

init(autoreset=True)

WORK_DIR = Path.cwd()


def mod_lmp_in(graph: Path, lmp_in: str = "in.lammps"):

    print(f"using graph: {graph}")

    lmp_in = (WORK_DIR / "in.lammps").read_text()
    lmp_in = re.sub(r"\.\./\.\./\.\./\S*", f"../../../{graph.name}", lmp_in)
    (WORK_DIR / "in.lammps").write_text(lmp_in)

    print("lammps input modified")


def vasp_recompute(
    *,
    lmp_binary: str,
    atom_style: str = "atomic",
    lmp_in: str = "in.lammps",
):

    vasp: List[Path]
    vasp = [
        d
        for d in WORK_DIR.rglob("*/")
        if d.is_dir()
        and d.parent.name == "vasp"
        and not (d.parent.parent / ".noEV").is_file()
    ]

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


def collect_lmp(collect_dirs: List[Path], lammpses: Tuple[str, ...], ace: bool):

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
                vol /= N
                en /= N
                if ace:  # weird shift that is present in CASTEP data
                    en += 157.72725320
                data.append([vol, en, stress])

            data = pd.DataFrame(data, columns=["volume", "energy", "stress"])
            data.sort_values(by=["volume"], inplace=True)
            data.to_csv(
                wd / "vol_stress.txt", sep=" ", float_format="%.6f", index=False
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

        dataf = pd.DataFrame(data, columns=["volume", "energy", "stress", "spg"])
        dataf.sort_values(by=["volume"], inplace=True)
        dataf.to_csv(wd / "vol_stress.txt", sep=" ", float_format="%.6f", index=False)


def run(args, graph: Optional[Path]):

    RECOMPUTE = args["recompute"]
    EQ = args["equation"]
    TRAIN_DIR = args["train_dir"]
    MTD_RUNS = args["mtds"]
    ABINIT_DIR = WORK_DIR if not args["abinit_dir"] else Path(args["abinit_dir"])

    if RECOMPUTE and args["potential"] is None and not graph:
        raise ValueError("Must input at least one graph if you want to recompute")

    collect_dirs = collect_data_dirs(ABINIT_DIR, reference=args["reference_structure"])
    assert len(collect_dirs) <= len(COLORS), (
        f"There is not enough colors to plot, "
        f"You are missing {len(collect_dirs) - len(COLORS)} more color(s)"
    )
    collect_vasp(collect_dirs)
    fig_ev, reference_ev, figm_ev, ax_ev = plot_abinit_ev(
        collect_dirs, EQ, shift_ref=args["shift_ev"]
    )
    fig_hp, reference_hp, figm_hp, ax_hp = plot_abinit_hp(
        collect_dirs, eos=EQ, fit=False, show_original_points=True
    )

    if graph or args["potential"] is not None:

        if args["potential"] == "deepmd":
            mod_lmp_in(graph)
            potential = "DEEPMD"
            labels = ("DPMD",)
        elif args["potential"] == "ace":
            potential = "ACE"
            labels = ("ACE",)
        elif args["potential"] == "tersoff":
            potential = "TERSOFF"
            labels = ("TERSOFF", )
        else:
            raise NotImplementedError("potential not implemented")

        # calculate nnp
        if RECOMPUTE:
            vasp_recompute(lmp_binary=args["lmp"])

        lammpses = ("lammps",)  # , "gc_lammps")

        collect_lmp(collect_dirs, lammpses, True if args["potential"] == "ace" else False)

        fig_ev = plot_predicted_ev(
            collect_dirs, EQ, lammpses, labels, fig_ev, reference_ev
        )

        fig_ev.update_layout(
            title=f"E-V diagram for {potential} potential vs Ab Initio calculations"
        )
        fig_hp = plot_predicted_hp(
            collect_dirs,
            lammpses,
            labels,
            fig_hp,
            reference_hp,
            eos=EQ,
            fit=False,
            show_original_points=False,
        )
        fig_hp.update_layout(
            title=f"H(p) plot for {potential} potential vs Ab Initio calculations"
        )
        # plot_mpl(collect_dirs)
    else:
        graph = Path.cwd() / "graph"
        fig_ev.update_layout(title="dataset E-V coverage")

    fig_ev.update_layout(
        xaxis_title=f"V [{ANGSTROM}{POW3} / atom]",
        yaxis_title="E [eV / atom]",
        # template="minimovka",
    )
    fig_hp.update_layout(
        xaxis_title=f"p [GPa]",
        yaxis_title=f"{DELTA}H [eV / atom], {collect_dirs[0].name} is reference",
        # template="minimovka",
    )

    #  matplotlib H(p)
    ax_hp.set(
        xlabel=f"p [GPa]",
        ylabel=f"{DELTA}H [eV / atom], {collect_dirs[0].name} is reference",
    )
    ax_hp.grid(linewidth=0.5)
    ax_hp.yaxis.set_minor_locator(AutoMinorLocator())
    ax_hp.xaxis.set_minor_locator(AutoMinorLocator())
    if args["x_span_hp"] is not None:
        ax_hp.set_xlim(args["x_span_hp"][0], args["x_span_hp"][1])
    if args["y_span_hp"] is not None:
        ax_hp.set_ylim(args["y_span_hp"][0], args["y_span_hp"][1])
    figm_hp.legend(loc="upper left", bbox_to_anchor=(0.25, 0.35), borderaxespad=0.0)

    # matplotlib E(V)
    ax_ev.set(
        xlabel=f"$V [\AA^3 / atom] $",
        ylabel=f"E [eV / atom]",
    )
    #ax_ev.grid(linewidth=0.5)
    ax_ev.yaxis.set_minor_locator(AutoMinorLocator())
    ax_ev.xaxis.set_minor_locator(AutoMinorLocator())
    if args["x_span_ev"] is not None:
        ax_ev.set_xlim(args["x_span_ev"][0], args["x_span_ev"][1])
    if args["y_span_ev"] is not None:
        ax_ev.set_ylim(args["y_span_ev"][0], args["y_span_ev"][1])

    if MTD_RUNS:
        plot_mtd_data(fig_ev, MTD_RUNS)

    if TRAIN_DIR:
        plot_coverage_data(fig_ev, ax_ev, TRAIN_DIR, args["graph_cache"], args["error_type"])

    figm_ev.legend(loc='center left', ncol=1, bbox_to_anchor=(1.0, 0.53), prop={'size': 8})

    if graph:
        basename = graph.stem
    elif args["potential"] == "ace":
        basename = "ACE"
    elif args["potential"] == "tersoff":
        basename = "TERSOFF"
    else:
        raise NotImplementedError("potential not implemented")

    filename = f"({basename}-{args['reference_structure']}).html"

    print(f"writing: {filename}")
    fig_ev.write_html(f"E-V{filename}", include_plotlyjs="cdn")
    fig_hp.write_html(f"H-p{filename}", include_plotlyjs="cdn")

    print(f"writing: {filename.replace('html', 'png')}")
    figm_hp.savefig(
        f"H-p{filename}".replace("html", "png"), dpi=500, bbox_inches="tight"
    )
    figm_ev.savefig(
        f"E-V{filename}".replace("html", "png"), dpi=500, bbox_inches="tight"
    )
    import matplotlib.pyplot as plt
    plt.show()


def compare_ev(args: dict):

    if args["graph"]:
        graphs = get_remote_files(args["graph"])
        for graph in graphs:
            print(f"analyzing for graph {graph}")
            print("----------------------------------------------------------")
            run(args, graph)
    else:
        run(args, None)
