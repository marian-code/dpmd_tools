from numpy.core.defchararray import title
import plotly.graph_objects as go
from pathlib import Path
from dpdata import LabeledSystem
from itertools import groupby
import numpy as np
from itertools import zip_longest
import plotly.express as px
from PIL import ImageColor

COLOR = px.colors.qualitative.Dark24


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_std(cache_dirs):

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
        continue
    print(d, "------------------------------")

    cache_dirs = [c for c in d.glob("cache*")]
    cache_dirs.sort(key=lambda x: int(x.name.split("_")[3][1]))
    iter_caches = groupby(cache_dirs, key=lambda x: int(x.name.split("_")[3][1]))
    mean = []
    std = []
    for key, group in iter_caches:
        cache_dirs = list(group)
        cache_dirs.sort(key=lambda x: int(x.stem.rsplit("_", 1)[-1]))
        if len(cache_dirs) > 4:
            cache_dirs = grouper(cache_dirs, 4)
            mm = []
            ss = []
            for i, cd in enumerate(cache_dirs, 1):
                print(f"{key}.{i}")
                m, s, size = get_std(cd)
                mm.append(m)
                ss.append(s)
            mean.append(mm)
            std.append(ss)
        else:
            print(key)
            m, s, size = get_std(cache_dirs)
            mean.append(m)
            std.append(s)

    force_stats[d.parent.name] = np.vstack((np.array(mean), np.array(std)))
    system_sizes[d.parent.name] = size
    if j > 1000:  # for testing
        break

m = max([s.shape[1] for s in force_stats.values()])

for name, stats in force_stats.items():
    diff = m - stats.shape[1]
    force_stats[name] = np.hstack((np.zeros((2, diff)), stats))

force_stats = dict(sorted(force_stats.items(), key=lambda x: x[0]))
system_sizes = {k: system_sizes[k] for k in force_stats.keys()}

fig = go.Figure()
all_data = []
all_error = []
for (name, stats), c in zip(force_stats.items(), COLOR):
    print(name, stats, system_sizes[name])

    x = []
    data = []
    error = []
    for i in range(stats.shape[1]):
        if isinstance(stats[0][i], list):
            n = len(stats[0][i])
            for j in range(0, n):
                x.append(i + 1 + j / n)
                data.append(stats[0][i][j])
                error.append(stats[1][i][j])
        else:
            x.append(i + 1)
            data.append(stats[0][i])
            error.append(stats[1][i])

    plot_one(fig, x, data, error, name, c, system_sizes[name]/10000)
    all_data.append(data)
    all_error.append(error)

plot_one(
    fig, x, np.mean(all_data, axis=0), np.mean(all_error, axis=0), "average", COLOR[-2], 2
)
plot_one(
    fig,
    x,
    np.average(all_data, weights=list(system_sizes.values()), axis=0),
    np.average(all_error, weights=list(system_sizes.values()), axis=0),
    "weighted average",
    COLOR[-1],
    2
)
fig.update_layout(
    title="Evolution of mean and std of prediction errors for sub-datasets",
    xaxis_title="Dataset append iteration (non-integer iterations correspond to "
    "different NN architectures in same iteration)",
    yaxis_title="Mean of max of forces prediction error in sub-dataset [eV/A]"
)

fig.write_html("dev_force_evol.html", include_plotlyjs="cdn")

