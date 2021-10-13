from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ase import units
from ase.eos import EquationOfState
import pandas as pd
from ase.spacegroup import Spacegroup
from . import COLORS


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