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

import argparse
import re
import shutil
import subprocess
import sys
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ase import Atoms, io
from ase.eos import EquationOfState
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ssh_utilities import Connection
from tqdm import tqdm

sys.path.append("/home/rynik/OneDrive/dizertacka/code/rw")
sys.path.append("/home/rynik/OneDrive/dizertacka/code/nnp")

from lammps import read_lammps_data, read_lammps_out
from plotly_theme_setter import *
from vasp import read_vasp_out


COLORS = ("black", "blue", "purple", "green", "red", "cyan", "goldenrod")
WORK_DIR = Path.cwd()


def input_parser() -> dict:

    p = argparse.ArgumentParser(
        description="run E-V plots check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-n", "--nnp", help="input dir with nnp potential, if "
                   "none is input then potential in ./nnp_model will be used",
                   default=None)
    p.add_argument("-g", "--graph", default=None, nargs="+",
                   help="use deepMD graph(s). Can also input graphs on remote "
                   "(server@/path/to/file). Wildcard '*' is also accepted.")
    p.add_argument("-r", "--recompute", default=False, action="store_true",
                   help="if false only collect previous results and don't "
                   "run lammps")
    p.add_argument("-e", "--equation", default="birchmurnaghan", type=str,
                   choices=("birchmurnaghan", "p3"), help="choose equation to "
                   "fit datapoints")
    p.add_argument("-t", "--train-dir", default=None, type=str, nargs="*",
                   help="input directories with data subdirs so data coverage "
                   "can be computed")
    p.add_argument("-m", "--mtds", default=None, type=str, nargs="*",
                   help="input paths to en_vol.npz files from MTD runs, can "
                   "be local(e.g. ../run/en_vol.npz) or remote"
                   "(e.g. host@/.../en_vol.npz")

    return vars(p.parse_args())


def mod_lmp_in(graph: Path, lmp_in: str = "in.lammps"):

    print(f"using graph: {graph}")

    lmp_in = (WORK_DIR / "in.lammps").read_text()
    lmp_in = re.sub(r"\.\./\.\./\.\./\S*", f"../../../{graph.name}", lmp_in)
    (WORK_DIR / "in.lammps").write_text(lmp_in)

    print("lammps input modified")


def vasp_recompute(atom_style: str = "atomic", lmp_in: str = "in.lammps"):

    vasp: List[Path]
    vasp = [d for d in WORK_DIR.rglob("*/")
            if d.is_dir() and d.parent.name == "vasp"]

    lmp_binary = shutil.which("lmp")
    print(f"using lmp binary: {lmp_binary}")

    vasp_job = tqdm(vasp, ncols=100, total=len(vasp))

    for v in vasp_job:

        a: Atoms = io.read(v / "OUTCAR")

        l = v.parent.parent / "lammps" / v.name
        l.mkdir(exist_ok=True, parents=True)

        vasp_job.set_description(f"Computing {l.parent.parent.name}/{l.name}")

        write_lammps_data(str(l / "data.in"), a, atom_style=atom_style,
                          force_skew=True)

        shutil.copy2(WORK_DIR / lmp_in, l)

        out = subprocess.run([lmp_binary, "-in", lmp_in], cwd=l,
                             capture_output=True, encoding="utf-8")
        
        try:
            out.check_returncode()
        except subprocess.CalledProcessError as e:
            vasp_job.write(e)
            vasp_job.write(out.stdout)


def collect_data_dirs():

    collect_dirs = [d for d in WORK_DIR.glob("*/") if d.is_dir()]

    # move Cubic diamond to the begining
    index = 0
    for i, c in enumerate(collect_dirs):
        if c.name == "cd":
            index = i

    temp = collect_dirs[index]
    del collect_dirs[index]
    collect_dirs.insert(0, temp)

    return collect_dirs


def collect_lmp(collect_dirs: List[Path], lammpses: Tuple[str, ...]):

    iter_dirs = tqdm(collect_dirs, ncols=100, total=len(collect_dirs))

    for cd in iter_dirs:

        iter_dirs.set_description(f"get lmp data: {cd.parent.parent.name}/"
                                  f"{cd.name}")

        for wd in [cd / l for l in lammpses]:

            dirs = [d for d in wd.glob("*/") if d.is_dir()]

            data = []

            for d in dirs:
                N = len(read_lammps_data(d / "data.out"))
                en, vol, stress = read_lammps_out(d / "log.lammps")[:3]
                stress /= 1000
                data.append([vol / N, (en / N), stress])

            data = np.array(data)

            np.savetxt(wd / "vol_stress.txt", data[data[:, -1].argsort()],
                       header="# Volume Energy stress", fmt="%.6f")


def collect_vasp(collect_dirs: List[Path]):

    iter_dirs = tqdm(collect_dirs, ncols=100, total=len(collect_dirs))

    for cd in iter_dirs:

        iter_dirs.set_description(f"get vasp data: {cd.parent.parent.name}/"
                                  f"{cd.name}")

        wd = cd / "vasp"

        dirs = [d for d in wd.glob("*/") if d.is_dir()]

        data = []

        for d in dirs:
            N = len(read(d / "POSCAR"))
            en, vol, stress = read_vasp_out(d / "OUTCAR")[:3]
            data.append([vol / N, en / N, stress])

        data = np.array(data)

        np.savetxt(wd / "vol_stress.txt", data[data[:, -1].argsort()],
                  header="# Volume Energy stress", fmt="%.6f")


def plot_mpl(collect_dirs: List[Path], eos: str, lammpses: Tuple[str, ...],
             labels: Tuple[str, ...]):

    for wd, c in zip(collect_dirs, COLORS[:len(collect_dirs)]):

        vasp_data = np.loadtxt(wd / "vasp" / "vol_stress.txt",
                            skiprows=1, unpack=True)

        vasp_state = EquationOfState(vasp_data[0], vasp_data[1], eos=eos)
        x, y = itemgetter(4, 5)(vasp_state.getplotdata())


        plt.scatter(vasp_data[0], vasp_data[1], s=25, c=c)
        plt.plot(x, y, label=f"{wd} - Ab initio (VASP)", color=c,
                linestyle="-", linewidth=2)

        for l, lab, ls in zip(lammpses, labels, ("", ":", "-.", "-")):
            lmps_data = np.loadtxt(wd / l / "vol_stress.txt",
                                skiprows=1, unpack=True)

            lmps_state = EquationOfState(lmps_data[0], lmps_data[1], eos=eos)
            x, y = itemgetter(4, 5)(lmps_state.getplotdata())
            plt.scatter(lmps_data[0], lmps_data[1], s=25, c=c)
            plt.plot(x, y, label=f"{wd} - {lab}", color=c,
                    linestyle=ls, linewidth=2)

    plt.xlabel("volume per atom")
    plt.ylabel("energy per atom")
    plt.title("E-V diagram for NNP, GAP and SNAP potentials vs "
            "Ab Initio calculations")

    plt.tight_layout()

    plt.legend()
    plt.show()


def plot_plotly(collect_dirs: List[Path], eos: str, lammpses: Tuple[str, ...],
                labels: Tuple[str, ...]) -> go.Figure:

    fig = go.Figure()

    for wd, c in zip(collect_dirs, COLORS[:len(collect_dirs)]):

        vasp_data = np.loadtxt(wd / "vasp" / "vol_stress.txt",
                            skiprows=1, unpack=True)

        vasp_state = EquationOfState(vasp_data[0], vasp_data[1], eos=eos)
        x, y = itemgetter(4, 5)(vasp_state.getplotdata())

        fig.add_trace(go.Scattergl(
            x=x, y=y, name=f"{wd.name} - Ab initio (VASP)",
            line=dict(color=c, width=3, dash="solid"), legendgroup=f"vasp_{c}"
        ))
        fig.add_trace(go.Scattergl(
            x=vasp_data[0], y=vasp_data[1],
            name=f"{wd.name} - Ab initio (VASP)", mode="markers",
            showlegend=False, line=dict(color=c, width=3),
            marker_size=25, legendgroup=f"vasp_{c}"
        ))

        for l, lab, ls in zip(lammpses, labels, ("dash", "dot", "dashdot", "-")):
            lmps_data = np.loadtxt(wd / l / "vol_stress.txt",
                                skiprows=1, unpack=True)

            lmps_state = EquationOfState(lmps_data[0], lmps_data[1], eos=eos)
            x, y = itemgetter(4, 5)(lmps_state.getplotdata())

            fig.add_trace(go.Scattergl(
                x=x, y=y, name=f"{wd.name} - {lab}",
                line=dict(color=c, width=3, dash=ls), legendgroup=c
            ))
            fig.add_trace(go.Scattergl(
                x=lmps_data[0], y=lmps_data[1],
                name=f"{wd.name} - {lab}", mode="markers",
                showlegend=False, legendgroup=c,
                marker_line_width=3, marker_symbol="circle-open",
                marker_size=25, line=dict(color=c, width=3)
            ))

    fig.update_layout(title="E-V diagram for DeepMd potential vs "
                            "Ab Initio calculations",
                    xaxis_title=f'V [{ANGSTROM}{POW3} / atom]', yaxis_title='E [eV / atom]',
                    template="minimovka")

    return fig


def get_coverage(data_dir: Path, box: str = "box.raw", En: str = "energy.raw"):

    dirs = [d for d in data_dir.glob("*/")
            if (d / box).is_file() and (d / En).is_file()]

    iter_dirs = tqdm(dirs, ncols=100, total=len(list(dirs)))

    data = {}
    for d in iter_dirs:
        iter_dirs.set_description(f"get coverage: {d.name}")
        n_atoms = len((d / "type.raw").read_text().splitlines())
        energies = np.loadtxt(d / "energy.raw") / n_atoms
        cells = np.loadtxt(d / "box.raw").reshape((-1, 3, 3))
        volumes = np.array([np.abs(np.linalg.det(c)) for c in cells]) / n_atoms

        name = d.name.replace("data_", "")
        data[name] = {"energy": energies, "volume": volumes}

    return dict(sorted(data.items()))


def get_mtd_runs(paths: List[str]):

    data = {}
    for p in paths:
        if "@" in p:
            host, p = p.split("@")
            print(f"loading {p} from {host}")
            name = Path(p).parent.name
            with Connection(host, quiet=True) as c:
                with c.builtins.open(p, "rb") as f:
                    temp = np.load(f)
                    data[name] = {"volume": temp["volume"], "energy": temp["energy"]}
                    #print(data[name]["volume"])
        else:
            print(f"loading {p} from local PC")
            name = Path(p).parent.name
            data[name] = np.load(p)

    return data


def run(args, graph: Path):
    RECOMPUTE = args["recompute"]
    EQ = args["equation"]
    TRAIN_DIR = args["train_dir"]
    MTD_RUNS = args["mtds"]

    if graph:
        mod_lmp_in(graph)

    # calculate nnp
    if RECOMPUTE:
        vasp_recompute()

    lammpses = ("lammps", )#, "gc_lammps")
    labels = ("DPMD", )

    collect_dirs = collect_data_dirs()
    collect_lmp(collect_dirs, lammpses)
    collect_vasp(collect_dirs)

    # plot_mpl(collect_dirs)
    fig = plot_plotly(collect_dirs, EQ, lammpses, labels)

    if MTD_RUNS:
        mtd_data = get_mtd_runs(MTD_RUNS)

        for name, mdata in mtd_data.items():
            fig.add_trace(go.Scattergl(
                x=mdata["volume"], y=mdata["energy"], mode='markers', name=name
            ))

    if TRAIN_DIR:
        for t in TRAIN_DIR:
            coverage_data = get_coverage(Path(t))

            for name, cdata in coverage_data.items():
                fig.add_trace(go.Scattergl(
                    x=cdata["volume"], y=cdata["energy"],
                    mode='markers', name=name
                ))


    fig.write_html(f"E-V({graph.stem}).html", include_plotlyjs="cdn")


def get_graphs(input_graphs: List[str]) -> List[Path]:

    graphs = []
    for graph_str in input_graphs:
        host_path = graph_str.split("@")

        if len(host_path) == 1:
            host = "local"
            path_str = host_path[0]
            local = True
        else:
            host, path_str = host_path
            local = False
        print(f"Getting graph from {host}")

        with Connection(host, quiet=True, local=local) as c:

            remote_path = c.pathlib.Path(path_str)
            remote_root = c.pathlib.Path(remote_path.root)
            remote_pattern = str(remote_path.relative_to(remote_path.root))
            remote_graphs = list(remote_root.glob(remote_pattern))

            for rg in remote_graphs:
                local_graph = Path.cwd() / rg.name
                try:
                    c.shutil.copy2(rg, local_graph, direction="get")
                except shutil.SameFileError:
                    pass
                graphs.append(local_graph)

    return graphs


if __name__ == "__main__":
    args = input_parser()

    graphs = get_graphs(args["graph"])

    for graph in graphs:
        print(f"analyzing for graph {graph}")
        print("--------------------------------------------------------------")
        run(args, graph)
