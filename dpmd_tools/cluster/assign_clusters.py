"""Select unique data from dataset based on K-Means clustering.

The distance metric is Cosine similarity and structures are represented by
Oganov fingerprints.

Only works for single element compounds.
"""

from collections import deque
from pathlib import Path
from typing import List, Tuple

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from colorama import Fore, init
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from atexit import register, unregister
from sys import exit
import signal


init(autoreset=True)
WORK_DIR = Path.cwd()


def ctrl_exit_handler(signal_received, frame):
    # Handle any cleanup here
    print("\nSIGINT or CTRL-C detected. Exiting gracefully")
    exit(0)


class KmeansRunner:
    """Run MiniBatch K-means clustering.

    References
    ----------
    https://scikit-learn.org/0.15/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans.predict
    https://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html
    """

    def __init__(
        self, fp_files: List[Path], passes: int, batch_size: int, n_clusters: int
    ) -> None:

        rng = np.random.RandomState(0)
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=rng,
            verbose=True,
            compute_labels=True,
            init="k-means++",
        )
        self.fp_files = fp_files
        self.passes = passes
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self._labels = None
        self.inertia = deque()

    def run_iter(self):

        for fp in self.fp_files:
            print(
                f"{Fore.GREEN}Training cluster representation from file: "
                f"{Fore.RESET}{fp.name}"
            )
            print(
                "convergence.html file will be ploted every 1000 iterations in run "
                "directory so you can asses convergence"
            )
            data = np.load(fp)
            n_samples = data.shape[0]

            job = tqdm(range(self.passes), ncols=130, total=self.passes)
            old_inertia = 1e23
            for i in job:

                # if we take all data in one batch there is no need to do this in
                # every loop pass
                if i == 0 or n_samples > self.batch_size:
                    idx = np.random.randint(n_samples, size=self.batch_size)
                    feed_data = data[idx, :]
                    # normalize to make cosine similarity equal to
                    # euclidean l2 distance measure
                    # https://stats.stackexchange.com/questions/72978/vector-space-model-cosine-similarity-vs-euclidean-distance
                    feed_data /= np.linalg.norm(feed_data, axis=1)[:, None]
                    # TODO maybe norm ?
                    # data -= np.mean(data, axis=0)
                    # data /= np.std(data, axis=0)
                self.kmeans.partial_fit(feed_data)
                self.inertia.append(self.kmeans.inertia_ / self.batch_size)
                job.set_description(
                    f"inertia: {self.kmeans.inertia_ / self.batch_size:.6f}"
                )
                if abs(old_inertia - self.kmeans.inertia_) <= 1e-6:
                    print(
                        " - job has converged early, clusters inertia in successive "
                        "passes match with accuracy 1e-6"
                    )
                    break

                if i % 1000 == 0:
                    self.plot_convergence()

    def run_all(self):

        data = np.vstack([np.load(fp) for fp in self.fp_files])
        # normalize
        data = np.linalg.norm(data, axis=1)[:, None]
        # fit
        self.kmeans.fit(data)

    @property
    def labels(self) -> np.ndarray:

        if self._labels is None:
            labels = []
            for fp in self.fp_files:
                labels.append(self.kmeans.predict(np.load(fp)))

            self._labels = np.concatenate(labels)

        return self._labels

    def plot_convergence(self):

        n = np.array(range(len(self.inertia)))
        inertia = np.array(self.inertia)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=n, y=inertia, mode="lines", name="cluster abs inertia")
        )
        fig.add_trace(
            go.Scatter(
                x=n[:-1],
                y=np.ediff1d(inertia),
                mode="lines",
                name="cluster abs inertia difference",
            )
        )
        fig.update_layout(
            title="Mini batch K-means convergence",
            xaxis_title="Iteration [N]",
            yaxis_title="Clusters inertia",
        )
        fig.write_html("convergence.html", include_plotlyjs="cdn")


def get_en_vol(path: Path) -> Tuple[np.ndarray, np.ndarray]:

    n_atoms = len((path / "type.raw").read_text().splitlines())
    energies = np.loadtxt(path / "energy.raw") / n_atoms
    boxes = np.loadtxt(path / "box.raw").reshape(-1, 3, 3)
    volumes = np.linalg.det(boxes) / n_atoms

    return energies, volumes


def plot_clusters(energies: np.ndarray, volumes: np.ndarray, labels: np.ndarray):

    colors = list(mcd.CSS4_COLORS.values())

    fig = go.Figure()
    for i in range(labels.max() + 1):
        idx = np.argwhere(labels == i).flatten()
        plt.scatter(volumes[idx], energies[idx], c=colors[i], label=str(i))
        fig.add_trace(
            go.Scattergl(
                x=volumes[idx],
                y=energies[idx],
                mode="markers",
                name=str(i),
                marker_color=colors[i],
            )
        )

    plt.legend()
    plt.savefig("clusters.png")

    fig.write_html("clusters.html", include_plotlyjs="cdn")


def assign_clusters(args: dict):

    print(f"{Fore.GREEN}Script was run with these arguments --------------------------")
    for arg, value in args.items():
        print(f" - {arg:20}: {value}")

    # find and sort files in dir
    fp_files = [f for f in WORK_DIR.glob("fingerprints_*.npy")]
    fp_files.sort(key=lambda x: int(x.stem.split("_")[1]))

    print(f"{Fore.GREEN}Found fingerprint files ----------------------------------")
    for fp in fp_files:
        print(f" - {fp.name}")

    # run kmeans algo
    kmeans = KmeansRunner(
        fp_files,
        int(args["passes"]),
        int(args["batch_size"]),
        int(args["n_clusters"]),
    )
    signal.signal(signal.SIGINT, ctrl_exit_handler)
    register(finalize, kmeans)
    kmeans.run_iter()
    unregister(finalize)
    finalize(kmeans)

def finalize(kmeans: KmeansRunner):

    kmeans.plot_convergence()

    energies, volumes = get_en_vol(WORK_DIR)

    plot_clusters(energies, volumes, kmeans.labels)
    np.savetxt(WORK_DIR / "clusters.raw", kmeans.labels, fmt="%3d")
