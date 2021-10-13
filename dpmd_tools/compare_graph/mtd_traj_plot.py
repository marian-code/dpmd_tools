from typing import List

import numpy as np
import plotly.graph_objects as go
from dpmd_tools.utils import get_remote_files


def get_mtd_runs(expand_paths: List[str]):

    paths = get_remote_files(expand_paths, remove_after=True, same_names=True)

    data = {}
    for p in paths:
        if len(p.suffixes) == 2:
            name = p.suffixes[0][1:]
        else:
            name = p.parent.name
        data[name] = np.load(p)

    return data


def plot_mtd_data(fig: go.Figure, paths: List[str]):
    mtd_data = get_mtd_runs(paths)

    for name, mdata in mtd_data.items():
        fig.add_trace(
            go.Scattergl(
                x=mdata["volume"], y=mdata["energy"], mode="markers", name=name
            )
        )
