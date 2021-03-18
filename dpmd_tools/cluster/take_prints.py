"""Select unique data from dataset based on K-Means clustering.

The distance metric is Cosine similarity and structures are represented by
Oganov fingerprints.

Only works for single element compounds.
"""

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.ga.ofp_comparator import OFPComparator
from colorama import Fore, init
from joblib import Parallel, delayed
from tqdm import tqdm

from dpmd_tools.readers.to_ase import load_npy_data, load_raw_data

init(autoreset=True)
WORK_DIR = Path.cwd()


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
            print(" - loading from raw")
            atoms = load_raw_data(path)
        else:
            print(" - loading from npy")
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

        print(f"{Fore.GREEN}Will now take fingerprints for dataset")
        for i, atoms in enumerate(self.atom_chunks, 1):
            print(f" - processing chunk {i}/{len(self.atom_chunks)}")

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
        print(f" - dumping {filename.name}")
        np.save(filename, data)

    def dump_settings(self, comparator_settings: dict):

        with (self.path / "fingerprint_settings.json").open("w") as f:
            json.dump(comparator_settings, f, allow_nan=True, indent=4)

    @staticmethod
    def _split(a: Sequence[Any], n: int) -> Iterator[List[Any]]:

        for i in range(0, len(a), n):
            yield a[i : i + n]

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


def take_prints(args: dict):

    print(f"{Fore.GREEN}Script was run with these arguments --------------------------")
    for arg, value in args.items():
        print(f" - {arg:20}: {value}")

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

    print(f"fingeprinting: {WORK_DIR.name}")
    fingerprints = FingerprintDataset(
        WORK_DIR,
        comparator,
        comparator_settings,
        batch_size=args["batch_size"],
        parallel=args["parallel"],
    )
    fingerprints.run()
