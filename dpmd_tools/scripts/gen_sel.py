from itertools import groupby, zip_longest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dpdata import LabeledSystem
from dpmd_tools.readers.to_dpdata import read_dpmd_raw
from dpmd_tools.to_deepmd import ApplyConstraint
from PIL import ImageColor

COLOR = px.colors.qualitative.Dark24 + px.colors.qualitative.Dark24
MODELS_PER_GEN = 4
ARGS = {
    "energy": (-4.5, -3.6),
    "volume": (12, 28.2),
    "pressure": (-1, 100),
    "per_atom": True,
}


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_ref_mask(cache_dirs, args: Dict[str, Any]):

    # get to the all data dir
    # we have: some_path/cache_<id>/
    # change to: some_path/all/Ge<XX>/
    ref_path = list((cache_dirs[0] / "../all").glob("*"))[0]
    reference = read_dpmd_raw(ref_path, force_iteration=0)[0]

    # load reference data and create a mask for it based on constraints
    # than use this mask to filter out undesired forces entries
    ac = ApplyConstraint(reference, False, lambda *args: args, False, False)
    if args["energy"]:
        ac.energy(bracket=args["energy"], per_atom=args["per_atom"])
    if args["volume"]:
        ac.volume(bracket=args["volume"], per_atom=args["per_atom"])
    if args["pressure"]:
        ac.pressure(bracket=args["pressure"])
    ac.apply()

    return reference.mask


def get_f_std(cache_dirs: List[Path], mask: Optional[np.ndarray]):

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

    # get only selected frames
    if mask is not None:
        # print(f_std_max.shape)
        f_std_max = f_std_max[mask > 0]
        # print(f_std_max.shape)

    return f_std_max.mean(), f_std_max.std(), len(f_std_max)


def get_e_std(cache_dirs: List[Path], mask: Optional[np.ndarray]):

    if None in cache_dirs:
        for c in cache_dirs:
            print(f" - {c}")
        raise FileNotFoundError("One or more expected cache dirs was not found")

    predictions = [LabeledSystem(c, fmt="deepmd/npy") for c in cache_dirs]

    # shape: (n_models, n_frames)
    energies = np.column_stack(
        [p.data["energies"] / p.get_natoms() for p in predictions]
    )

    e_std = np.std(energies, axis=1)

    # get only selected frames
    if mask is not None:
        # print(e_std.shape)
        e_std = e_std[mask > 0]
        # print(e_std.shape)

    return e_std.mean(), e_std.std(), len(e_std)


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

    """
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
    """


def read_errors(
    dirs: List[Path],
    error_reader: Callable[
        [List[Path], Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray, int]
    ]
) -> Tuple[pd.DataFrame, Dict[str, int]]:

    stats = {}
    system_sizes = {}
    for j, d in enumerate(dirs):

        d = d / "deepmd_data"
        if not d.is_dir() or len(list(d.glob("cache*"))) == 0:
            print(f"{str(d):60} ... skipping")
            continue
        print(f"{str(d):60} ... ", end="")

        # list all cache directories
        cache_dirs = [c for c in d.glob("cache*")]

        # sort according to generation: e.g. cache_ge_all_s1_4.pb --> generation 1
        cache_dirs.sort(key=lambda x: int(x.name.split("_")[3][1]))
        # get mask for structures to sort out ones constrined by energ, volume or stress
        mask = get_ref_mask(cache_dirs, ARGS)
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
                    m, s, size = error_reader(cd, mask)
                    mean.append(m)
                    std.append(s)
                    generation.append(number)
            else:
                m, s, size = error_reader(cache_dirs, mask)
                mean.append(m)
                std.append(s)
                generation.append(float(key))

        print("OK")

        # create dictionary, eych entry contains dataframe with mean and std for
        # each generation
        stats[d.parent.name] = pd.DataFrame(
            {"mean": mean, "std": std}, index=generation
        )

        # record system size so the weighted average can be computed
        system_sizes[d.parent.name] = size
        if j > 100:  # for testing
            break

    # make one multiindex dataframe from distionary of individual dataframes
    stats = pd.concat(stats.values(), axis=1, keys=stats.keys())
    stats.fillna(method="backfill", inplace=True)

    # ensure system sizes dict which is used to compute weighted average is
    # ordered in the same way as the dataframe
    system_sizes = {
        name: system_sizes[name] for name in unique(stats.columns.get_level_values(0))
    }

    return stats, system_sizes


def plot_stats(stats: pd.DataFrame, system_sizes: Dict[str, int], *, graph_type: str):

    shift = -0.01
    shift_step = 0.02 / stats.shape[1] * 2
    # plot individual systems prediction error, iterate over system names in dataframe
    # which are sored in level 0 column labels
    fig = go.Figure()
    for (name, stat), c in zip(stats.groupby(level=0, axis="columns"), COLOR):
        s = stat[name]
        plot_one(
            fig,
            s.index.to_numpy() + shift,
            s["mean"].to_numpy(),
            s["std"].to_numpy(),
            name,
            c,
            system_sizes[name] / 10000,
        )
        shift += shift_step

    # use cross section locator to select all columns with appropriate quantity
    # these are on sublevel 1. then do a men on these columns (column -> axis==1)
    plot_one(
        fig,
        stats.index,
        stats.xs("mean", level=1, axis=1).mean(axis=1),
        stats.xs("std", level=1, axis=1).mean(axis=1),
        "average",
        COLOR[-2],
        2,
    )
    # again use cross section locator to select appropriate columns then multiply each
    # column by respective weight, after that sum and devide by weight total
    ss = system_sizes.values()

    """
    print(stats)
    print("------------------------------")
    print(ss)
    print("------------------------------")
    print(stats.xs("mean", level=1, axis=1))
    print("---------------")
    print(stats.xs("mean", level=1, axis=1).multiply(ss))
    print("--------------------------------")
    print(stats.xs("mean", level=1, axis=1).multiply(ss).sum(axis=1))
    print("--------------------------------")
    print(stats.xs("mean", level=1, axis=1).multiply(ss).sum(axis=1) / sum(ss))
    print("--------------------------------")
    """

    plot_one(
        fig,
        stats.index,
        stats.xs("mean", level=1, axis=1).multiply(ss).sum(axis=1) / sum(ss),
        stats.xs("std", level=1, axis=1).multiply(ss).sum(axis=1) / sum(ss),
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

    if graph_type == "force":
        fig.update_layout(
            yaxis_title="Mean of max of forces prediction error in sub-dataset [eV/A]",
        )

        fig.write_html("dev_force_evol.html", include_plotlyjs="cdn")
        print("saved dev_force_evol.html")
    elif graph_type == "energy":
        fig.update_layout(
            yaxis_title="mean energy prediction error in sub-dataset [A]",
        )

        fig.write_html("dev_energy_evol.html", include_plotlyjs="cdn")
        print("saved dev_energy_evol.html")


dirs = list(Path.cwd().glob("*"))
#dirs = list(Path.cwd().glob("*0GPa*"))
#dirs = list(Path.cwd().glob("*md*600*"))
dirs.sort(key=lambda x: x.name)

#print("ploting energy deviations")
#stats, system_size = read_errors(dirs, get_e_std)
#plot_stats(stats, system_size, graph_type="energy")

print("ploting force deviations")
stats, system_size = read_errors(dirs, get_f_std)
print(stats)
plot_stats(stats, system_size, graph_type="force")
