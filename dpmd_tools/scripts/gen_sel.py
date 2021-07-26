from numpy.core.defchararray import array, title
import plotly.graph_objects as go
from pathlib import Path
from dpdata import LabeledSystem
from itertools import groupby
import numpy as np
from itertools import zip_longest
import plotly.express as px
from PIL import ImageColor
import pandas as pd

COLOR = px.colors.qualitative.Dark24 + px.colors.qualitative.Dark24
MODELS_PER_GEN = 4


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_std(cache_dirs):

    if None in cache_dirs:
        for c in cache_dirs:
            print(f" - {c}")
        raise FileNotFoundError("One or more expected cache dirs was not found")

    predictions = [LabeledSystem(c, fmt="deepmd/npy") for c in cache_dirs]

    # shape: (n_models, n_frames, n_atoms, 3)
    forces = np.stack([p.data["forces"] for p in predictions])

    # shape: (n_frames, n_atoms, 3)
    f_std = np.std(forces, axis=0)

    # shape: (n_frames, n_atoms)
    f_std_size = np.linalg.norm(f_std, axis=2)

    # shape: (n_frames, )
    f_std_max = np.max(f_std_size, axis=1)

    return f_std_max.mean(), f_std_max.std(), len(f_std_max)


def plot_one(fig: go.Figure, x, data, error, name, color, linewidth):

    fig.add_trace(
        go.Scatter(
            x=x,
            y=data,
            error_y=dict(type="data", array=error, visible=True),
            name=name,
            line=dict(color=color, width=linewidth),
        )
    )

    y_upper = (np.array(data) + np.array(error)).tolist()
    y_lower = (np.array(data) - np.array(error)).tolist()
    rgb = ImageColor.getcolor(color, "RGB")

    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper + y_lower[::-1],  # upper, then lower reversed
            fill="toself",
            fillcolor="rgba({},{},{},0.2)".format(*rgb),
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="continuous",
            legendgroup=1,
        )
    )


force_stats = {}
system_sizes = {}
for j, d in enumerate(Path.cwd().glob("*")):

    d = d / "deepmd_data"
    if not d.is_dir() or len(list(d.glob("cache*"))) == 0:
        print(f"{str(d):60} ... skipping")
        continue
    print(f"{str(d):60} ... OK")

    # list all cache directories
    cache_dirs = [c for c in d.glob("cache*")]
    # sort according to generation: e.g. cache_ge_all_s1_4.pb --> generation 1
    cache_dirs.sort(key=lambda x: int(x.name.split("_")[3][1]))
    # group by generation number
    iter_caches = groupby(cache_dirs, key=lambda x: int(x.name.split("_")[3][1]))
    mean = []
    std = []
    generation = []
    # key is generation number, and group contains cachce dirs
    for key, group in iter_caches:
        cache_dirs = list(group)
        # sort cache dirs
        cache_dirs.sort(key=lambda x: int(x.stem.rsplit("_", 1)[-1]))
        # iterate in chunks defined by constant
        if len(cache_dirs) > MODELS_PER_GEN:
            cache_dirs = grouper(cache_dirs, MODELS_PER_GEN)
            for i, cd in enumerate(cache_dirs):
                number = float(f"{key}.{i}")
                m, s, size = get_std(cd)
                mean.append(m)
                std.append(s)
                generation.append(number)
        else:
            m, s, size = get_std(cache_dirs)
            mean.append(m)
            std.append(s)
            generation.append(float(key))

    # create dictionary, eych entry contains dataframe with mean and std for
    # each generation
    force_stats[d.parent.name] = pd.DataFrame(
        {"mean": mean, "std": std}, index=generation
    )

    # record system size so the weighted average can be computed
    system_sizes[d.parent.name] = size
    if j > 1000:  # for testing
        break

# make one multiindex dataframe from distionary of individual dataframes
force_stats = pd.concat(force_stats.values(), axis=1, keys=force_stats.keys())
force_stats.fillna(method="backfill", inplace=True)

# ensure system sizes dict which is used to compute weighted average is
# ordered in the same way as the dataframe
system_sizes = {
    name: system_sizes[name] for name in unique(force_stats.columns.get_level_values(0))
}

# plot individual systems prediction error, iterate over system names in dataframe
# which are sored in level 0 column labels
fig = go.Figure()
for (name, stats), c in zip(force_stats.groupby(level=0, axis="columns"), COLOR):
    stats = stats[name]
    plot_one(
        fig,
        stats.index,
        stats["mean"].to_numpy(),
        stats["std"].to_numpy(),
        name,
        c,
        system_sizes[name] / 10000,
    )

# use cross section locator to select all columns with appropriate quantity
# these are on sublevel 1. then do a men on these columns (column -> axis==1)
plot_one(
    fig,
    force_stats.index,
    force_stats.xs("mean", level=1, axis=1).mean(axis=1),
    force_stats.xs("std", level=1, axis=1).mean(axis=1),
    "average",
    COLOR[-2],
    2,
)
# again use cross section locator to select appropriate columns then multiply each
# column by respective weight, after that sum and devide by weight total
ss = system_sizes.values()
plot_one(
    fig,
    force_stats.index,
    force_stats.xs("mean", level=1, axis=1).multiply(ss).sum(axis=1) / sum(ss),
    force_stats.xs("std", level=1, axis=1).multiply(ss).sum(axis=1) / sum(ss),
    "weighted average",
    COLOR[-1],
    2,
)
fig.update_layout(
    title="Evolution of mean and std of prediction errors for sub-datasets",
    xaxis_title="Dataset append iteration (non-integer iterations correspond to "
    "different NN architectures in same iteration)",
    yaxis_title="Mean of max of forces prediction error in sub-dataset [eV/A]",
)

fig.write_html("dev_force_evol.html", include_plotlyjs="cdn")
