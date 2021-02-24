from typing import Any, Callable, List, NoReturn, Tuple

import numpy as np
from tqdm import tqdm

from dpmd_tools.data import LabeledSystemMask
from functools import wraps


def check_bracket(function: Callable) -> Callable:
    @wraps(function)
    def decorator(self: "ApplyConstraint", **kwargs) -> Callable:

        bracket = kwargs["bracket"]
        if bracket[1] < bracket[0]:
            raise ValueError(f"{function.__name__} bounds are in incorrect order")
        else:
            return function(self, **kwargs)

    return decorator


class ApplyConstraint:
    """Apply constraints to dataset and filter out unsatisfactory structures.

    Each method must concatenate to `sel_indices` - the indices of structures
    it found that satisfy the imposed conditions.
    """

    _predictions = None

    def __init__(
        self,
        name: str,
        system: LabeledSystemMask,
        max_structures: int,
        use_prints: bool,
        lprint: Callable[[Any], NoReturn],
    ) -> None:

        self.lprint = lprint
        self.system = system
        self.max_structures = max_structures
        self.use_prints = use_prints
        self.sel_indices = np.arange(self.system.get_nframes(), dtype=int)

        self.lprint(f"filtering system {name}")

    def get_predictions(self, graphs: List[str]) -> List[LabeledSystemMask]:

        if not self._predictions:
            self.lprint("computing model predictions")
            self._predictions = []
            job = tqdm(enumerate(graphs, 1), ncols=100, total=len(graphs))
            for i, g in job:
                job.set_description(f"graph {i}/{len(graphs)}")
                self._predictions.append(self.system.predict(g))
        return self._predictions

    def _record_select(self, sel_indices: np.ndarray):
        self.sel_indices = np.intersect1d(self.sel_indices, sel_indices)

    def _where(self, cond) -> np.ndarray:
        return np.argwhere(cond & (self.system.mask == 0)).flatten()

    def _limit_structures(self):
        self.lprint(f"applying max structures ({self.max_structures}) limit")

        # if upper limit on number of selected structures is specified
        if self.max_structures is not None:
            over_limit = len(self.sel_indices) - self.max_structures

            # if the number of selected structures is not over limit, return
            if over_limit <= 0:
                return

            self.lprint(
                f"will delete {over_limit} additional structures, because "
                "max structures argument is specified"
            )

            if self.use_prints and self.system.data["clusters"] is not None:
                self.lprint(
                    "will use fingerprint based clustering to select the most diverse "
                    "structures"
                )
                count = np.unique(self.system.clusters)[1]
                # get probability we will encounter cluster 'i'
                prob = count / self.system.get_nframes()
                # reverse probability, so the least frequent structures
                # are most probable
                prob = 1 - prob
                # map probability to whole array
                probability = np.empty(self.system.get_nframes())
                for i, _ in enumerate(count):
                    probability[self.system.clusters == i] = prob[i]

                # norm probability to 1
                probability = probability / probability.sum()
                # select
                self.sel_indices = np.random.choice(
                    self.sel_indices, self.max_structures, p=probability, replace=False
                )

            else:
                if not self.system.data["clusters"] is None:
                    self.lprint("cannot use fingerprints, clusters.raw file is missing")

                # randomly select indices, the number that is selected is such that
                # taht after deleting these number of selected indices will fit into
                # the max_structures limit
                self.sel_indices = np.random.choice(
                    self.sel_indices, self.max_structures, replace=False
                )

    @check_bracket
    def energy(self, *, bracket: Tuple[float, float], per_atom: bool = True):
        self.lprint(f"based on energy{' per atom' if per_atom else ''}")

        energies = self.system.data["energies"]
        if per_atom:
            energies = energies.copy() / self.system.get_natoms()

        s = self._where((bracket[0] < energies) & (energies < bracket[1]))

        self.lprint(f"selected {len(s)} frames as a result of energy constraints")

        self._record_select(s)

    def every(self, *, n_th: int):
        self.lprint(f"based on: take every {n_th} frame criterion")

        indices = np.arange(self.system.get_nframes())

        #  because we have masked system some structures might not be available
        # so we try to shift the choose indices by one in each loop and choose
        # the result that gives most structures
        s = []
        for i in range(n_th):
            s.append(self._where((indices % n_th == i)))

        s = sorted(s, key=lambda x: len(x), reverse=True)[0]

        self.lprint(
            f"selected {len(s)} frames as a result of take every n-th frame constraints"
        )

        self._record_select(s)

    @check_bracket
    def volume(self, *, bracket: Tuple[float, float], per_atom: bool = True):
        self.lprint(f"based on volume{' per atom' if per_atom else ''}")

        # this an array of square matrices, linalg det auto cancualtes det for
        # every sub matrix that is square
        volumes = np.linalg.det(self.system.data["cells"])
        if per_atom:
            volumes /= self.system.get_natoms()

        s = self._where((bracket[0] < volumes) & (volumes < bracket[1]))

        self.lprint(f"selected {len(s)} frames as a result of volume constraints")

        self._record_select(s)

    @check_bracket
    def dev_e(
        self,
        *,
        graphs: List[str],
        bracket: Tuple[float, float],
        std_method: bool = False,
        per_atom: bool = True,
    ):
        """Select which labeled structures should be added to dataset.

        This method is usefull when you already have DFT data labels without
        any prior selection and want to decide which of them should be added to
        the dataset based on dataset predictions for them.

        See Also
        --------
        :func:`dpgen.simplify.simplify.post_model_devi`
        """
        self.lprint("based on energy std")

        predictions = self.get_predictions(graphs)

        # shape: (n_models, n_frames)
        energies = np.stack([p.data["energies"] for p in predictions])
        if per_atom:
            energies /= self.system.get_natoms()

        if std_method:
            e_std = energies.std(axis=0)
        else:
            reference = self.system.data["energies"] / self.system.get_natoms()
            # make column vector of reference DFT data
            reference = np.atleast_2d(reference).T
            e_std = np.sqrt(np.mean(np.power(abs(energies - reference), 2), axis=0))

        # save for plotting
        self.system.data["energies_std"] = e_std

        s = self._where((bracket[0] < e_std) & (e_std < bracket[1]))

        self.lprint(f"selected {len(s)} frames as a result of energy std constraints")

        self._record_select(s)

    @check_bracket
    def dev_f(
        self,
        *,
        graphs: List[str],
        bracket: Tuple[float, float],
        std_method: bool = False,
    ):
        """Select which labeled structures should be added to dataset.

        This method is usefull when you already have DFT data labels without
        any prior selection and want to decide which of them should be added to
        the dataset based on dataset predictions for them.

        See Also
        --------
        :func:`dpgen.simplify.simplify.post_model_devi`
        """
        self.lprint(f"based on max atom force std")

        predictions = self.get_predictions(graphs)

        # shape: (n_models, n_frames, n_atoms, 3)
        forces = np.stack([p.data["forces"] for p in predictions])

        # shape: (n_frames, n_atoms, 3)
        if std_method:
            f_std = np.std(axis=0)
        else:
            reference = self.system.data["forces"]
            f_std = np.sqrt(np.mean(np.power(abs(forces - reference), 2), axis=0))

        # shape: (n_fames, n_atoms)
        f_std_size = np.linalg.norm(f_std, axis=2)

        # shape: (n_frames, )
        f_std_max = np.max(f_std_size, axis=1)

        # save for plotting
        self.system.data["forces_std_max"] = f_std_max

        s = self._where((bracket[0] < f_std_max) & (f_std_max < bracket[1]))

        self.lprint(
            f"selected {len(s)} frames as a result of max forces std constraints"
        )

        self._record_select(s)

    def apply(self) -> LabeledSystemMask:
        self.lprint(f"selecting {len(self.sel_indices)} frames")

        self._limit_structures()

        # create used structures mask
        mask = np.zeros(self.system.get_nframes(), dtype=int)
        mask[self.sel_indices] = 1
        # append to existing mask
        self.system.append_mask(mask)

        # create filtered subsystem
        sub_indices = self.system.get_subsystem_indices()
        sub_system = self.system.sub_system(sub_indices)

        # copy custom attributes
        for attr in ["energies_std", "forces_std_max"]:
            try:
                sub_system.data[attr] = self.system.data[attr][sub_indices]
            except KeyError:
                print(f"Cannot copy key: {attr} to subsystem")

        return sub_system
