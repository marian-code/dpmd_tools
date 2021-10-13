import plotly.graph_objects as go
from ssh_utilities import Connection
from pathlib import Path
from ssh_utilities import SSHPath
from typing import List, Iterator, Dict, Literal, Optional, Union

import numpy as np
from tqdm import tqdm
from fnmatch import fnmatch
from dpdata import LabeledSystem


def get_coverage(
    data_dir: Union[Path, SSHPath],
    types: str = "type.raw",
    graph_glob: Optional[str] = None,
    error_type: Literal["energy", "force"] = "energy"
):

    print("finding directories")
    # TODO a lot of hard coding
    dirs = [
        d
        for d in tqdm(data_dir.glob("**"), ncols=100)
        if (d / types).is_file() and "cache" not in str(d) and "for_train" not in str(d)
        and "unused" not in str(d)
    ]

    if isinstance(data_dir, SSHPath) and graph_glob:
        raise TypeError("cannot read deviation data from remote file system!")

    for i, d in enumerate(dirs):
        print(f"{i:>2}. {d}")

    iter_dirs: Iterator[Path] = tqdm(dirs, ncols=100, total=len(list(dirs)))

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for d in iter_dirs:
        if "minima" in str(d):
            continue
        iter_dirs.set_description(f"get coverage: {d.name}")  # type: ignore
        n_atoms = len((d / "type.raw").read_text().splitlines())
        sets = sorted(d.glob("set.*"), key=lambda x: x.name.split(".")[1])

        if sets:
            energies = (
                np.concatenate([np.load((s / "energy.npy").open("rb")) for s in sets])
                / n_atoms
            )
            cells = np.vstack(
                [np.load((s / "box.npy").open("rb")) for s in sets]
            ).reshape((-1, 3, 3))
        else:
            energies = np.loadtxt((d / "energy.raw").open("rb")) / n_atoms
            cells = np.loadtxt((d / "box.raw").open("rb")).reshape(-1, 3, 3)

        volumes = np.abs(np.linalg.det(cells)) / n_atoms

        # * we want to extract here the "mc_cd" name
        # data/md_cd/...raw
        if len(d.relative_to(data_dir).parts) == 1:
            name = d.name.replace("data_", "")
            cache_root = None
        # data/md_cd/deepmd_data/all/Ge136/...raw
        elif len(d.relative_to(data_dir).parts) == 4 and "all" in str(d):
            name = d.parent.parent.parent.name
            cache_root = d.parent.parent
        # data/md_cd/Ge136/...raw
        else:
            name = d.parent.name.replace("data_", "")
            cache_root = None

        if not cache_root:
            raise FileNotFoundError(
                "we do not know where to look for prediction cache "
                "in this data structure"
            )
        cache_dirs = [
            d for d in cache_root.glob("**") if fnmatch(d.name, graph_glob)
        ]

        if error_type == "energy":
            error = get_e_std(cache_dirs)
        elif error_type == "force":
            error = get_f_std(cache_dirs)

        if name in data:
            data[name]["energy"] = np.concatenate((data[name]["energy"], energies))
            data[name]["volume"] = np.concatenate((data[name]["volume"], volumes))
            data[name]["error"] = np.concatenate((data[name]["error"], error))
        else:
            data[name] = {"energy": energies, "volume": volumes, "error": error}

    return dict(sorted(data.items()))


def get_f_std(cache_dirs: List[Path]):

    predictions = [LabeledSystem(c, fmt="deepmd/npy") for c in cache_dirs]

    # shape: (n_models, n_frames, n_atoms, 3)
    forces = np.stack([p.data["forces"] for p in predictions])

    # shape: (n_frames, n_atoms, 3)
    f_std = np.std(forces, axis=0)

    # shape: (n_frames, n_atoms)
    f_std_size = np.linalg.norm(f_std, axis=2)

    # shape: (n_frames, )
    f_std_max = np.max(f_std_size, axis=1)

    return f_std_max


def get_e_std(cache_dirs: List[Path]):

    predictions = [LabeledSystem(c, fmt="deepmd/npy") for c in cache_dirs]

    # shape: (n_models, n_frames)
    energies = np.column_stack(
        [p.data["energies"] / p.get_natoms() for p in predictions]
    )

    return np.std(energies, axis=1)



def plot_coverage_data(
    fig: go.Figure,
    train_dirs: List[str],
    graph_glob: Optional[str],
    error_type: Literal["energy", "force"]
):

    for t in train_dirs:
        if "@" in t:
            server, t = t.split("@")
            local = False
        else:
            server = "LOCAL"
            local = True

        with Connection(server, quiet=True, local=local) as c:
            #print(c)
            path = c.pathlib.Path(t)
            coverage_data = get_coverage(
                path, graph_glob=graph_glob, error_type=error_type
            )

        show = True
        for name, cdata in coverage_data.items():

            if graph_glob:
                str_labels = []
                for index, err in enumerate(cdata["error"]):
                    str_labels.append(
                        f"index: {index}<br>{error_type} error: {err:.4f}"
                    )
                fig.add_trace(
                    go.Scattergl(
                        x=cdata["volume"],
                        y=cdata["energy"],
                        mode="markers",
                        name=name,
                        text=str_labels,
                        marker=dict(
                            color=cdata["error"],  # set color equal to a variable
                            colorscale='Viridis',  # one of plotly colorscales
                            showscale=show
                        )
                    )
                )
            else:
                str_labels = []
                for index, vol in enumerate(cdata["volume"]):
                    str_labels.append(
                        f"index: {index}"
                    )
                fig.add_trace(
                    go.Scattergl(
                        x=cdata["volume"],
                        y=cdata["energy"],
                        mode="markers",
                        name=name,
                        text=str_labels,
                    )
                )

            show = False

    fig.update_layout(legend_orientation="h")