"""Select unique data from dataset based on K-Means clustering.

The distance metric is Cosine similarity and structures are represented by
Oganov fingerprints.

Only works for single element compounds.
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dpmd_tools.readers import load_npy_data, load_raw_data

WORK_DIR = Path.cwd()


def input_parser():

    p = argparse.ArgumentParser(
        description="select data based on fingerprints, selection mode uses "
        "Mini Batch K-Means algorithm with online training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp = p.add_subparsers(
        dest="action",
        help="Select running mode, either "
        "take fingerprints or filter dat absed on already "
        "computed fingerprints",
    )

    prints = sp.add_parser("take-prints")
    prints.add_argument(
        "-bs",
        "--batch_size",
        default=1e6,
        type=int,
        help="Size of chunks that will be used to save "
        "fingerprint data, to avoid memory overflow",
    )
    prints.add_argument(
        "-p",
        "--parallel",
        default=False,
        action="store_true",
        help="whether to run fingerprinting in parallel, usually there is speedup "
        "benefit up to an order of magnitude",
    )

    select = sp.add_parser("select")
    select.add_argument(
        "-p",
        "--passes",
        default=100,
        type=int,
        help="number ov dataset passes of MiniBatch K-means online learning loop.",
    )
    select.add_argument(
        "-bs",
        "--batch_size",
        default=int(1e6),
        type=int,
        help="Data will be iterativelly loaded from files "
        "specified in the fingerprinting phase, and from "
        "those random batches will be chosen of size "
        "specified by this argument",
    )
    select.add_argument(
        "-nf",
        "--n-from-cluster",
        default=100,
        type=int,
        help="number of random samples to select from each "
        "cluster. Data will be devided to two folders: selected and the rest",
    )
    select.add_argument(
        "-nc",
        "--n-clusters",
        default=100,
        type=int,
        help="target number of clusters for K-Means algorithm",
    )

    both = sp.add_parser("both")
    both.add_argument(
        "-p",
        "--passes",
        default=100,
        type=int,
        help="number ov dataset passes of MiniBatch K-means online learning loop.",
    )
    both.add_argument(
        "-bs",
        "--batch_size",
        default=int(1e6),
        type=int,
        help="Data will be iterativelly loaded from files "
        "specified in the fingerprinting phase, and from "
        "those random batches will be chosen of size "
        "specified by this argument",
    )
    both.add_argument(
        "-nf",
        "--n-from-cluster",
        default=100,
        type=int,
        help="number of random samples to select from each "
        "cluster. Data will be devided to two folders: selected and the rest",
    )
    both.add_argument(
        "-nc",
        "--n-clusters",
        default=100,
        type=int,
        help="target number of clusters for K-Means algorithm",
    )
    both.add_argument(
        "-pa",
        "--parallel",
        default=False,
        action="store_true",
        help="whether to run "
        "fingerprinting in parallel, usually there is speedup "
        "benefit up to an order of magnitude",
    )

    return vars(p.parse_args())


class FingerprintDataset:
    def __init__(
        self,
        path: Path,
        comparator: OFPComparator,
        comparator_settings: dict,
        batch_size: Optional[int] = None,
        parallel: bool = False,
    ) -> None:

        sets = [f for f in path.glob("*") if (f / "box.npy").is_file()]
        if len(sets) == 0:
            print("loading from raw")
            atoms = load_raw_data(path)
        else:
            print("loading from npy")
            atoms = load_npy_data(path)

        if batch_size and batch_size < len(atoms):
            self.atom_chunks = list(self._split(atoms, batch_size))
        else:
            self.atom_chunks = [atoms]

        self.comparator = comparator
        self.path = path
        self.parallel = parallel

        self.dump_settings(comparator_settings)

    def run(self):

        for i, atoms in enumerate(self.atom_chunks, 1):
            print(f"processing chunk {i}/{len(self.atom_chunks)}")

            job = tqdm(atoms, ncols=100, total=len(atoms))

            if self.parallel:
                data = self._run_parallel(job)
            else:
                data = self._run_serial(job)

            fingerprints = np.vstack(data)

            self._dump_data(fingerprints, i)

    def _run_serial(self, job: Iterator[Atoms]) -> List[np.ndarray]:

        data = []
        for a in job:
            self._take_fingerprints(a, self.comparator)

        return data

    def _run_parallel(self, job: Iterator[Atoms]) -> List[np.ndarray]:

        pool = Parallel(n_jobs=12, backend="loky")
        exec = delayed(self._take_fingerprints)
        return pool(exec(a, self.comparator) for a in job)

    def _dump_data(self, data: np.ndarray, chunk: int):

        filename = self.path / f"fingerprints_{chunk}.npy"
        print(f"dumping {filename.name}")
        np.save(filename, data)

    def dump_settings(self, comparator_settings: dict):

        with (self.path / "fingerprint_settings.json").open("w") as f:
            json.dump(comparator_settings, f, allow_nan=True, indent=4)

    @staticmethod
    def _split(a: Sequence[Any], n: int) -> Iterator[List[Any]]:

        for i in range(0, len(a), n):
            yield a[i: i + n]

    @staticmethod
    def _take_fingerprints(a: Atoms, comparator: OFPComparator) -> np.ndarray:
        """Use suplied comparator to take fingerprint.

        Warnings
        --------
        Works only for single element compounds!!!
        """
        fp, _ = comparator._take_fingerprints(a)
        Z = a.get_atomic_numbers()[0]

        return fp[(Z, Z)]


def get_data_dirs(path: Path) -> List[Path]:

    paths = []
    for p in path.glob("*/"):
        if (p / "box.raw").is_file() and (p / "coord.raw").is_file():
            paths.append(p)

    return paths


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
            print(f"training data from file {fp.name}")
            data = np.load(fp)

            job = tqdm(range(self.passes), ncols=130, total=self.passes)
            for i in job:

                idx = np.random.randint(data.shape[0], size=self.batch_size)
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
                    f"pass {i + 1}/{self.passes}, inertia: "
                    f"{self.kmeans.inertia_ / self.batch_size:.5f}"
                )

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
                name="cluster abs inertia",
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
    for i in range(100):
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


def main():

    args = input_parser()
    print(args)

    comparator_settings = {
        "n_top": None,
        "dE": 1,
        "cos_dist_max": 5e-3,
        "rcut": 10,  # 10
        "binwidth": 0.08,  # 0.08
        "pbc": True,
        "maxdims": None,
        "sigma": 0.02,  # 0.02
        "nsigma": 4,  # 4
    }

    comparator = OFPComparator(**comparator_settings)

    if args["action"] in ("take-prints", "both"):

        print(f"fingeprinting: {WORK_DIR.name}")
        fingerprints = FingerprintDataset(
            WORK_DIR,
            comparator,
            comparator_settings,
            batch_size=args["batch_size"],
            parallel=args["parallel"],
        )
        fingerprints.run()

    if args["action"] in ("select", "both"):

        # find and sort files in dir
        fp_files = [f for f in WORK_DIR.glob("fingerprints_*.npy")]
        fp_files.sort(key=lambda x: int(x.stem.split("_")[1]))

        # print out
        print("fingerprint files:")
        for fp in fp_files:
            print(fp.name)

        # run kmeans algo
        kmeans = KmeansRunner(
            fp_files,
            int(args["passes"]),
            int(args["batch_size"]),
            int(args["n_clusters"]),
        )
        kmeans.run_iter()
        kmeans.plot_convergence()

        energies, volumes = get_en_vol(WORK_DIR)

        plot_clusters(energies, volumes, kmeans.labels)
        np.savetxt(WORK_DIR / "clusters.raw", kmeans.labels, fmt="%3d")


if __name__ == "__main__":
    main()
