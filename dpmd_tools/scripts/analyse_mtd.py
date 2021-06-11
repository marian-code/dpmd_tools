import re
import shutil
import subprocess
import tarfile
import warnings
from copy import copy
from os import fspath
from pathlib import Path

import matplotlib as mpl
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
from joblib import Parallel, delayed
from tqdm import tqdm

mpl.use("Agg")
mpl.rcParams["agg.path.chunksize"] = 10000

WORK_DIR = Path.cwd()
FES_DIR = WORK_DIR / "fes_files"
OUT_DIR = WORK_DIR / f"{WORK_DIR.name}_out"
KJ_TO_EV = 1.0 / 96.0
GPA = 0.0001
# Parameters of the system
CELL_CONSTANT = 1
# Parameters for plots
SZ = 15
LW = 1
WD = 10
HG = 5


class PlotlyPlots:
    def __init__(self) -> None:
        self.plots = []

    def get_plot(self, x, ydata, title, xlabel, ylabel, labels=None, save_name=None):

        # matlplotlib
        fig = plt.figure(figsize=(WD, HG))
        plt.rc("xtick", labelsize=SZ)
        plt.rc("ytick", labelsize=SZ)

        if labels:
            lines = []
            for y, l in zip(ydata, labels):
                lines.append(plt.plot(x, y, linewidth=LW, label=l)[0])

            plt.legend(handles=lines, loc="best", fontsize=SZ)
        else:
            plt.plot(x, ydata, color="k", linewidth=1.0)

        plt.xlabel(xlabel, fontsize=SZ)
        plt.ylabel(ylabel, fontsize=SZ)

        if not save_name:
            save_name = f"{title}.png"
        else:
            plt.title(title)

        plt.savefig(OUT_DIR / save_name, dpi=300, format="png")
        plt.close()

        # plotly
        fig = go.Figure()

        if labels:
            for y, l in zip(ydata, labels):
                fig.add_trace(go.Scattergl(x=x, y=y, name=l))
        else:
            fig.add_trace(go.Scattergl(x=x, y=ydata))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            # template="minimovka",
            # yaxis=dict(scaleanchor="x", scaleratio=1),
            # xaxis=dict(constrain="domain"),
            font=dict(family="Courier New, monospace", size=SZ),
        )
        fig.write_html(
            str((OUT_DIR / f"{title}.html").resolve()),
            include_plotlyjs="cdn",
            include_mathjax="cdn",
        )

        self.plots.append(
            {
                "title": title,
                "html": po.plot(
                    fig, include_plotlyjs=False, output_type="div", auto_open=False
                ),
            }
        )

    def get_animated_plot(
        self,
        data,
        title,
        xlabel,
        ylabel,
        zlabel=None,
        heatmap=False,
        embed=True,
        cn_range=None,
        v_range=None,
    ):

        # https://plotly.com/python/sliders/
        # https://stackoverflow.com/questions/58948973/python-change-custom-control-values-in-plotly

        fig = go.Figure()

        # Add traces, one for each slider step
        for i, d in enumerate(data):
            if zlabel and heatmap:
                fig.add_trace(
                    go.Heatmap(
                        visible=False,
                        name=d["t"],
                        x=d["x"],
                        y=d["y"],
                        z=d["z"],
                        zsmooth="fast",
                    )
                )
            elif zlabel and not heatmap:
                fig.add_trace(
                    go.Surface(
                        contours={
                            "z": {
                                "show": True,
                                "start": d["z"].min(),
                                "end": d["z"].max(),
                                "size": 0.05,
                            }
                        },
                        visible=False,
                        name=d["t"],
                        x=d["x"],
                        y=d["y"],
                        z=d["z"],
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(visible=False, name=d["t"], x=d["x"], y=d["y"])
                )

        # Make 10th trace visible
        fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [
            dict(
                active=10,
                currentvalue={"prefix": "MTD time: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(
            sliders=sliders,
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            # template="minimovka",
            # yaxis=dict(scaleanchor="x", scaleratio=1),
            # xaxis=dict(constrain="domain"),
            font=dict(family="Courier New, monospace", size=SZ),
            scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel),
        )

        if cn_range and v_range:
            fig.update_layout(xaxis=dict(range=cn_range), yaxis=dict(range=v_range))

        if cn_range and not v_range:
            fig.update_layout(yaxis=dict(range=cn_range))

        if v_range and not cn_range:
            fig.update_layout(yaxis=dict(range=v_range))

        if not zlabel:
            y_min = min([d["y"].min() for d in data])
            y_max = max([d["y"].max() for d in data])
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))
        else:
            fig.update_layout(
                scene=dict(zaxis=dict(range=[d["z"].min(), d["z"].max()]))
            )

        for i, d in enumerate(data):
            fig["layout"]["sliders"][0]["steps"][i]["label"] = f"{d['t']}[ps]"

        fig.write_html(
            str((OUT_DIR / f"{title}.html").resolve()),
            include_plotlyjs="cdn",
            include_mathjax="cdn",
        )

        if embed:
            self.plots.append(
                {
                    "title": title,
                    "html": po.plot(
                        fig, include_plotlyjs=False, output_type="div", auto_open=False
                    ),
                }
            )
        else:
            pass


def make_html(figs):

    # make basic header and include plotly js
    html = (
        "<head>"
        "   <!-- Plotly.js -->"
        '   <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        '   <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"></script>'
        "<style>"
        "h2 {"
        "  color: #1c87c9;"
        "}"
        "h2:target {"
        "  color: white;"
        "  background: #1c87c9;"
        "}"
        "</style>"
        "</head>"
        "<body>"
    )

    # add table of contents
    html += '<h1 id="toc">Table of contents</h1>' "<ol>"

    for i, f in enumerate(figs):
        html += "<li>" f'   <a href="#fig{i}">{f["title"]}</a>' "</li>"

    html += "</ol>"

    # add the figures
    for i, f in enumerate(figs):
        html += (
            f'<h1 id="fig{i}">{f["title"]}</h1>'
            "<p> Go to the "
            '    <a href="#toc">table of contents</a>.'
            "</p>"
            f'{f["html"]}'
        )

    html += "</body>"

    return html


def get_fes(
    index,
    fes,
    min_CN,
    max_CN,
    min_V,
    max_V,
    palette,
    stride,
    grid_CN,
    grid_V,
    temperature,
    n_atoms,
):

    dataset = pd.read_table(
        fes,
        header=0,
        skiprows=8,
        delimiter=r"\s+",
        names=["CN", "Volume", "F"],
        usecols=[0, 1, 2],
    )
    dataset = dataset.sort_values(["CN", "Volume"])

    free_energy = dataset["F"].to_numpy()
    free_energy *= KJ_TO_EV / n_atoms
    free_energy = free_energy.reshape((Nbin_CN, Nbin_V))

    plt.imshow(
        free_energy.T,
        origin="lower",
        aspect="auto",
        cmap=palette,
        vmin=np.min(free_energy),
        vmax=np.max(free_energy),
        extent=[min_CN, max_CN, min_V, max_V],
    )
    plt.colorbar()

    plt.xlabel(r"Coordination number $[1]$")
    plt.ylabel(r"Volume/atom $[A^3]$")

    plt.title(
        r"Free energy/atom $[eV]$ - at $t = "
        + "{:01.3f}".format(stride * (index + 1))
        + r"\,ps$"
    )

    plt.savefig(OUT_DIR / f"FES_{index}.png", dpi=350, format="png")
    plt.close()

    map_data = {
        "x": grid_CN,
        "y": grid_V,
        "z": free_energy.T,
        "t": "{:01.3f}".format(stride * (index + 1)),
    }

    free_energy -= np.min(free_energy)
    probability = np.exp(-free_energy / temperature)

    probability_CN = np.sum(probability, axis=1)
    probability_V = np.sum(probability, axis=0)

    free_energy_CN = -temperature * np.log(probability_CN)
    free_energy_V = -temperature * np.log(probability_V)

    cn_data = {
        "x": grid_CN,
        "y": free_energy_CN,
        "t": "{:01.3f}".format(stride * (index + 1)),
    }
    v_data = {
        "x": grid_V,
        "y": free_energy_V,
        "t": "{:01.3f}".format(stride * (index + 1)),
    }

    return map_data, cn_data, v_data


def load_data(lmp_out: str, plumed_in: str, plumed_out: str):
    def _sort_lmp(path: Path):
        filename = path.name

        try:
            return int(filename.split(".")[-1])
        except ValueError as e:
            print(f"file {filename} error: {e}")
            return 1e6

    def _find_thermo_start(path: Path, get_lines: bool = True):

        lines = ""
        skip_min_output = False
        with path.open("r") as f:
            for i, line in enumerate(f):
                if get_lines:
                    lines += line
                if line.startswith("minimize"):
                    skip_min_output = True
                if line.startswith("Step"):
                    if skip_min_output:
                        skip_min_output = False
                    else:
                        upper_limit = i + 1
                        break
            else:
                raise RuntimeError("Cannot find start of thermo output")

        if get_lines:
            return upper_limit, lines
        else:
            return upper_limit

    N_ATOMS_PATTERN = re.compile(r"\s*reading atoms\s*...\s*\n\s*(\d+)\s*atoms")
    N_ATOMS_PATTERN_R = re.compile(r"\s+(\d+)\s*atoms\n")
    THERMO_STRIDE = re.compile(r"\s*thermo\s*(\d+)")
    HEADER = re.compile(r"\s*thermo_style\s*custom\s*.*\n")
    PLMD_STRIDE = re.compile(r"PRINT\s*ARG=\*\s*FILE=COLVAR\s*STRIDE=(\d+)")

    # Read number of atoms from log.lammps
    # Find upper most of the table
    upper_limit, lines = _find_thermo_start(WORK_DIR / lmp_out, get_lines=True)

    lmp_files = [f for f in (WORK_DIR).glob(f"{lmp_out}*")]

    if len(lmp_files) == 1:
        print("It appears that there was only one run and no restarts")
    else:
        print(f"It appears that run was continued {len(lmp_files) - 1} times")
        print(f"assumed file order is:")

        lmp_files.sort(key=_sort_lmp)
        for i, f in enumerate(lmp_files):
            print(f"{i:>2d}: {f}")

        print(
            "file output frequency cannot have changed between the runs, "
            "otherwise results will be wrong!"
        )

    try:
        # normal file
        n_atoms = int(N_ATOMS_PATTERN.findall(lines)[0])
    except IndexError:
        # restart file
        n_atoms = int(N_ATOMS_PATTERN_R.findall(lines)[0])
    lmp_stride = int(THERMO_STRIDE.findall(lines)[0])
    lmp_header = HEADER.findall(lines)[0].split()[2:]

    plmd_stride = int(PLMD_STRIDE.findall((WORK_DIR / plumed_in).read_text())[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = []
        for lf in lmp_files:

            upper_limit = _find_thermo_start(lf, get_lines=False)

            data.append(
                np.genfromtxt(
                    lf,
                    skip_header=upper_limit,
                    comments="#",
                    skip_footer=0,
                    invalid_raise=False,
                )
            )

        cn = np.genfromtxt(
            WORK_DIR / plumed_out,
            skip_header=1,
            invalid_raise=False,
            usecols=(3,),
            unpack=True,
        )

    if len(data) == 1:
        data = data[0]
    else:
        print(data[0][:, 0])

        for i, _ in enumerate(data):
            if i < len(data) - 1:
                print(
                    "slicing data so the previous run ends where the " "next continues"
                )
                data[i] = data[i][data[i][:, 0] < data[i + 1][(0, 0)]]
                print(
                    "adjusting timestep because after restart lammps always "
                    "starts from 0"
                )
                data[i + 1][:, 1] += data[i][(-1, 1)] + data[i + 1][(1, 1)]

        print("stacking data")
        data = np.vstack(data)

    if plmd_stride == lmp_stride:
        print(f"plumed stride: {plmd_stride} is equal to lmp: {lmp_stride}")
        common_stride = plmd_stride
    elif plmd_stride % lmp_stride == 0:
        print(f"plumed stride: {plmd_stride} is greater than lmp: {lmp_stride}")
        common_stride = int(plmd_stride / lmp_stride)
        data = data[::common_stride]
    elif lmp_stride % plmd_stride == 0:
        print(f"lmp stride: {lmp_stride} is greater than plumed: {plmd_stride}")
        common_stride = int(lmp_stride / plmd_stride)
        cn = cn[::common_stride]
    else:
        common_stride = np.lcm(plmd_stride, lmp_stride)
        print(
            f"plumed stride: {plmd_stride} is not a multiple of "
            f"lmp stride: {lmp_stride} or vice-versa lowest common "
            f"multiple is: {common_stride}"
        )
        data = data[::common_stride]
        cn = cn[::common_stride]

    print("slicing...")
    print(len(data), len(cn), common_stride)

    assert abs(len(data) - len(cn)) < common_stride, "data sliced wrong!"

    # if run was ended prematurely data can have different length, that is ok
    # as long as the dif is smalled than common_stride
    if len(data) > len(cn):
        print(
            f"plummed COLVAR has last {len(data) - len(cn)} rows missing "
            f"compared to log.lammps!"
        )
        data = data[: len(cn)]
    elif len(data) < len(cn):
        print(
            f"log.lammps has last {len(cn) - len(data)} rows missing "
            f"compared to plumed COLVAR!"
        )
        cn = cn[: len(data)]

    df = pd.DataFrame(data=data, columns=lmp_header)
    df["cn"] = cn

    return df, n_atoms


def analyse_mtd(*, ev_only, **kwargs):

    # remove bck files
    print("removing backup files")
    for f in WORK_DIR.glob("bck.*"):
        f.unlink()

    print("loading data to memory")
    d, n_atoms = load_data("log.lammps", "plumed.dat", "COLVAR")
    print("Data loaded into memory")
    print("Plotting all quantities ...")

    # Aliases to the quantities
    step = d["step"]
    time = d["time"]
    temp = d["temp"]

    pressure = d["press"] * GPA
    target_pressure = np.mean(pressure)

    volume = d["vol"] / n_atoms
    pot_eng = d["pe"] / n_atoms
    kin_eng = d["ke"] / n_atoms

    enthalpy_with_ke = d["enthalpy"] / n_atoms
    enthalpy_calculated = pot_eng + target_pressure * volume * 6.324e-4

    cell_x = d["lx"]
    cell_y = d["ly"]
    cell_z = d["lz"]

    Pxx = d["pxx"] * GPA
    Pyy = d["pyy"] * GPA
    Pzz = d["pzz"] * GPA
    Pxy = d["pxy"] * GPA
    Pxz = d["pxz"] * GPA
    Pyz = d["pyz"] * GPA

    cell_a = d["cella"]
    cell_b = d["cellb"]
    cell_c = d["cellc"]

    cell_alpha = d["cellalpha"]
    cell_beta = d["cellbeta"]
    cell_gamma = d["cellgamma"]

    cn = d["cn"]

    np.savez_compressed(
        WORK_DIR / "en_vol.npz",
        volume=volume,
        energy=(pot_eng + kin_eng),
        timesteps=time,
    )
    print("saved energy volume file")

    if ev_only:
        return
    else:
        FES_DIR.mkdir(exist_ok=True)
        OUT_DIR.mkdir(exist_ok=True)

    pp = PlotlyPlots()  # plotly plots

    # Total system energy -- conserved quantity
    pp.get_plot(
        time,
        (pot_eng + kin_eng),
        "00_extended_energy",
        r"$t [ps]$",
        r"$Energy/atom [eV]$",
    )

    # Temperature
    pp.get_plot(time, temp, "01_temperature", r"$t$ $[ps]$", r"$Temperature [K]$")

    # Potential energy
    pp.get_plot(
        time, pot_eng, "02_pot_energy", r"$t [ps]$", r"$Potential energy/atom [eV]$"
    )

    # Enthalpy
    pp.get_plot(
        time,
        enthalpy_with_ke,
        "07_enthalpy",
        r"$t [ps]$",
        r"$Enthalpy (with KE)/atom [eV]$",
    )

    # Enthalpy "calculated by hand"
    pp.get_plot(
        time,
        enthalpy_calculated,
        "08_enthalpy_calculated",
        r"$t [ps]$",
        r"$Enthalpy/atom [eV]$",
    )

    # Volume
    pp.get_plot(time, volume, "10_volume", r"$t [ps]$", r"$Volume/atom [A^3]$")

    # Potential Energy - Volume projection [WITHOUT]
    pp.get_plot(
        volume,
        pot_eng,
        "12_pot_energy_volume_without.png",
        r"$Volume/atom [A^3]$",
        r"$Potential energy/atom [eV]$",
    )

    # Cell angles
    pp.get_plot(
        time,
        (cell_alpha, cell_beta, cell_gamma),
        "13_cell_angles",
        r"$t [ps]$",
        r"$Cell angles [^\circ]$",
        labels=(r"$\alpha$", r"$\beta$", r"$\gamma$"),
    )

    # Pressure
    pp.get_plot(time, pressure, "14_pressure", r"$t [ps]$", r"$Pressure [GPa]$")

    # Stress tensor
    pp.get_plot(
        time,
        (Pxx, Pyy, Pzz),
        "15_pressure_tensor_components_diagonal",
        r"$t [ps]$",
        r"$Pressure tensor components [GPa]$",
        labels=(r"xx", r"yy", r"zz"),
    )

    # Off diagonal pressure tensor components
    pp.get_plot(
        time,
        (Pxy, Pxz, Pyz),
        "16_pressure_tensor_components_off_diagonal",
        r"$t [ps]$",
        r"$Pressure tensor components [GPa]$",
        labels=(r"xy", r"xz", r"yz"),
    )

    # Cell size
    pp.get_plot(
        time,
        (cell_a / CELL_CONSTANT, cell_b / CELL_CONSTANT, cell_c / CELL_CONSTANT),
        "17_cell_size",
        r"$t [ps]$",
        r"$Unit cell size [A]$",
        labels=("a", "b", "c"),
    )

    # Plot COLVAR
    # CV in time
    pp.get_plot(time, cn, "18_CN_in_time", r"$Time [ps]$", r"Coordination number")

    # CV vs volume
    pp.get_plot(
        volume, cn, "19_CN_volume", r"$Volume/atom [eV]$", r"Coordination number"
    )

    # CV vs pot_eng
    pp.get_plot(
        pot_eng,
        cn,
        "20_CN_pot_eng",
        r"$Potential energy/atom [eV]$",
        r"Coordination number",
    )

    # CN vs H
    pp.get_plot(
        cn,
        enthalpy_calculated,
        "21_CN_enthalpy",
        r"Coordination number",
        r"$Enthalpy/atom [eV]$",
    )

    # Plot Gibss free energy surface
    # Number of free energy cuts at different times
    FREQUENCY = 500  # ps
    PACE = 1  # ps
    stride = int(FREQUENCY / PACE)

    fes_files = [f for f in FES_DIR.glob("fes_*.dat")]
    if not fes_files:
        print("Summing HILLS using PLUMED...")

        try:
            out = subprocess.run(
                ["plumed", "sum_hills", "--hills", "HILLS", "--stride", str(stride)],
                capture_output=True,
                encoding="utf-8",
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(e)
            print(out.stderr)
        else:
            print("Success")
            fes_files = []
            for f in WORK_DIR.glob("fes_*.dat"):
                shutil.move(fspath(f), FES_DIR)
                fes_files.append(FES_DIR / f.name)
    else:
        print("HILLS already summed")

    # Init
    temperature = 2.494 * KJ_TO_EV  # 300 K in eV
    palette = copy(plt.get_cmap("YlOrRd"))
    palette.set_bad("w", 1.0)
    palette.set_over("w", 1.0)
    palette.set_under("w", 1.0)

    # Plot FES in different times

    fes_files.sort(key=lambda x: int(re.findall(r"\d+", x.name)[0]))

    # Find minimal and maximal values
    # this is extracted from header
    df = pd.read_table(
        FES_DIR / "fes_0.dat",
        header=0,
        names=("data",),
        nrows=6,
        delimiter=r"\s+",
        usecols=(3,),
        skiprows=(4, 8, 9),
    )

    min_CN, max_CN, Nbin_CN, min_V, max_V, Nbin_V = df["data"].tolist()
    Nbin_CN = int(Nbin_CN)
    Nbin_V = int(Nbin_V)

    grid_CN = np.linspace(min_CN, max_CN, Nbin_CN)
    grid_V = np.linspace(min_V, max_V, Nbin_V)

    print("analyzing fes files")

    job_data = tqdm(enumerate(fes_files), ncols=100, total=len(fes_files))
    pool = Parallel(n_jobs=10, backend="loky")
    runner = delayed(get_fes)
    data = pool(
        runner(
            index,
            fes,
            min_CN,
            max_CN,
            min_V,
            max_V,
            palette,
            stride,
            grid_CN,
            grid_V,
            temperature,
            n_atoms,
        )
        for index, fes in job_data
    )

    v_data = []
    cn_data = []
    map_data = []
    for d in data:
        map_data.append(d[0])
        cn_data.append(d[1])
        v_data.append(d[2])

    print("saving FES htmls")

    # number of points decreases quadratically!
    TAKE_EVERY = 3
    print(f"subsampling map data, taking every {TAKE_EVERY} frame")
    for m in map_data:
        m["x"] = m["x"][tuple([slice(None, None, TAKE_EVERY)] * m["x"].ndim)]
        m["y"] = m["y"][tuple([slice(None, None, TAKE_EVERY)] * m["y"].ndim)]
        m["z"] = m["z"][tuple([slice(None, None, TAKE_EVERY)] * m["z"].ndim)]

    cn_range = [min_CN, max_CN]
    v_range = [min_V, max_V]

    pp.get_animated_plot(
        cn_data,
        r"FES_CN_Free energy-atom - CN[eV]",
        r"$Coordination number [1]$",
        r"$Free energy/atom [eV]$",
        cn_range=cn_range,
    )
    pp.get_animated_plot(
        v_data,
        r"FES_CN_Free energy-atom - V[eV]",
        r"$Volume/atom [A^3]$",
        r"$Free energy/atom [eV]$",
        v_range=v_range,
    )
    pp.get_animated_plot(
        map_data,
        r"FES_CN_Free energy-atom (CN, V) 3D [eV]",
        r"$Coordination number [1]$",
        r"$Volume/atom [A^3]$",
        zlabel=r"$Free energy/atom [eV]$",
        embed=False,
        cn_range=cn_range,
        v_range=v_range,
    )
    pp.get_animated_plot(
        map_data,
        r"FES_CN_Free energy-atom (CN, V) Heatmap [eV]",
        r"$Coordination number [1]$",
        r"$Volume/atom [A^3]$",
        zlabel=r"$Free energy/atom [eV]$",
        heatmap=True,
        embed=False,
        cn_range=cn_range,
        v_range=v_range,
    )

    print("saving complete html")
    (OUT_DIR / "results.html").write_text(make_html(pp.plots))

    print("saving all results to tar file")
    with tarfile.open(f"{OUT_DIR.name}.tar.gz", "w:gz") as tar:
        tar.add(OUT_DIR, arcname=OUT_DIR.name)
