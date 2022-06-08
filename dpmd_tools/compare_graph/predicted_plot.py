from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from ase import units
from ase.eos import EquationOfState
from colorama import Fore
import pandas as pd

from . import COLORS


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


def plot_predicted_hp(
    collect_dirs: List[Path],
    lammpses: Tuple[str, ...],
    labels: Tuple[str, ...],
    fig: go.Figure,
    reference: np.poly1d,
    *,
    eos: str,
    fit: bool,
    show_original_points=True,
) -> go.Figure:

    for wd, c in zip(collect_dirs, COLORS[: len(collect_dirs)]):
        for l, lab, ls in zip(lammpses, labels, ("dash", "dot", "dashdot", "-")):
            df = pd.read_table(wd / l / "vol_stress.txt", sep=" ")

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
            if show_original_points:
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
            lmps_data = pd.read_table(wd / l / "vol_stress.txt", sep=" ")

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
