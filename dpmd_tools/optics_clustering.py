"""methods for OPTICS clustering.

"""

from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import scipy
from joblib import Parallel, delayed
from sklearn.cluster import OPTICS
from tqdm import tqdm

from .data import convert_size

WORK_DIR = Path.cwd()


def cosine_distance(fp1: np.ndarray, fp2: np.ndarray):
    """Returns the cosine distance from two fingerprints.
    It also needs information about the number of atoms from
    each element, which is included in "typedic".

    Adapter from `ase.ga`

    Warnings
    --------
    Works only for single element compounds!!!
    See original ase implementations for details

    See also
    --------
    :class:`ase.ga.ofp_comparator.OFPComparator`
    """
    # calculating the fingerprint norms:
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)

    # calculating the distance:
    distance = np.sum(fp1 * fp2) / (norm1 * norm2)

    return 0.5 * (1 - distance)


class All2All:
    """COmpute all to all distance matrix for OPTICS clustering.
    
    Or any other clustering algorithm.
    memmap is for large datasets that will not fit to memory, keep in mind
    that all-to-all matrix will have (n_structures)^2 * size
    """

    def __init__(self, metrics: Callable[[np.ndarray, np.ndarray], float],
                 fingerprints: np.ndarray, memmap: bool,
                 mat_file: Optional[Path] = None) -> None:

        self.metrics = metrics
        self.memmap = memmap
        self.dim = len(fingerprints)
        self.indices = combinations(range(self.dim), 2)
        self.fp = fingerprints

        if memmap:
            if not mat_file:
                raise ValueError("must specify mat_file if memmap is True")
            else:
                self.mem_array = np.memmap(mat_file, dtype="float32", mode="w+",
                                           shape=(self.dim, self.dim))
                # fill diagonal
                for i in range(self.dim):
                    self.mem_array[i, i] = 0

    def iterate_chunks(self, iterable: Iterable, batch_size: int = 1) -> Iterator:

        buffer = deque()
        for i in tqdm(iterable, ncols=100, total=scipy.special.comb(self.dim, 2)):
            if len(buffer) >= batch_size:
                yield buffer
                buffer.clear()
            else:
                buffer.append(i)

        return buffer

    def distance_chunk(self, indices_pairs: List[Tuple[int, int]]):

        for i, j in indices_pairs:
            d = self.metrics(self.fp[i], self.fp[j])
            self.mem_array[i, j] = d
            self.mem_array[j, i] = d

    def compute(self):

        pool = Parallel(n_jobs=12, backend="loky")
        exec = delayed(self.distance_chunk)

        pool(exec(ind) for ind in self.iterate_chunks(self.indices))

        return self.mem_array


if __name__ == "__main__":

    # compute fingerprint with data selector first
    fp_files = [f for f in WORK_DIR.glob("fingerprints_*.npy")]
    fp_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    fp = np.vstack([np.load(fp) for fp in fp_files])

    a2a = All2All(cosine_distance, fp, memmap=True,
                  mat_file=WORK_DIR / "dist_amt.npy")

    distances = a2a.compute()
    print(distances)
    print(distances.nbytes / len(distances)**2)
    print(convert_size(distances.nbytes))
    print(np.count_nonzero(distances))

    samples = OPTICS(metric="precomputed").fit_predict(distances)
    print(samples)
