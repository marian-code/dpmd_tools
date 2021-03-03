"""Houses class that takes care of frame filtering for datasets construction."""

from dpmd_tools.compare_graph import WORK_DIR
from typing import Any, Callable, List, NoReturn, Optional, Tuple, Union

import numpy as np

from dpmd_tools.data import LabeledSystemMask, LabeledSystem
from functools import wraps


def check_bracket(function: Callable) -> Callable:
    """Check if bracketing interval was inputin correct order."""

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
        max_structures: Optional[Union[str, int]],
        use_prints: bool,
        lprint: Callable[[Any], NoReturn],
        append: bool = True,
        cache_predictions: bool = False,
    ) -> None:

        self.lprint = lprint
        self.system = system
        self.max_structures = max_structures
        self.cache_predictions = cache_predictions
        self.use_prints = use_prints
        self.sel_indices = np.arange(self.system.get_nframes(), dtype=int)
        self.append = append

        self.lprint(f"filtering system {name}")

    def get_predictions(self, graphs: List[str]) -> List[LabeledSystemMask]:

        if not self._predictions:
            self.lprint("computing model predictions")
            self._predictions = []
            for i, g in enumerate(graphs, 1):
                self.lprint(f"graph {i}/{len(graphs)}")

                # load/save cached predictions to avoid recomputing when we need to run
                # again. This happens quite often since we need to guess bracket
                # intervals and there is not knowing beforehand how many structures this
                # will yield
                cache_dir = WORK_DIR / f"cache_{g.name}"
                if self.cache_predictions:
                    if cache_dir.is_dir():
                        self.lprint(
                            f"found cached predictions: "
                            f"{cache_dir.relative_to(WORK_DIR)}"
                        )
                        system = LabeledSystem(cache_dir, fmt="deepmd/npy")
                    else:
                        self.lprint(
                            f"could not find cached predictions: "
                            f"{cache_dir.relative_to(WORK_DIR)}"
                        )
                        system = self.system.predict(g)
                else:
                    system = self.system.predict(g)

                self._predictions.append(system)

                if not cache_dir.is_dir():
                    self.lprint(
                        f"saving predictions to {cache_dir.relative_to(WORK_DIR)}"
                    )
                    system.to_deepmd_npy(cache_dir)
        return self._predictions

    def _record_select(self, sel_indices: np.ndarray):
        self.sel_indices = np.intersect1d(self.sel_indices, sel_indices)

    def _where(self, cond) -> np.ndarray:
        return np.argwhere(cond & (self.system.mask == 0)).flatten()

    def _limit_structures(self):

        if "%" in self.max_structures:
            max_s = int(
                self.system.get_nframes() *
                (float(self.max_structures.replace("%", "")) / 100)
            )
            self.lprint(f"{self.max_structures} portion equals {max_s} frames")
        else:
            max_s = int(self.max_structures)

        over_limit = len(self.sel_indices) - max_s

        # if the number of selected structures is not over limit, return
        if over_limit <= 0:
            return

        self.lprint(
            f"will delete {over_limit} structures from selection, because "
            "max structures argument is specified"
        )

        if self.use_prints and self.system.data["clusters"] is not None:
            self.lprint(
                "will use fingerprint based clustering to select the most diverse "
                "structures"
            )
            count = np.unique(self.system.clusters, return_counts=True)[1]
            # get probability we will encounter cluster 'i'
            prob = count / self.system.get_nframes()
            # reverse probability, so the least frequent structures
            # are most probable
            prob = 1 - prob
            # map probability to whole array
            probability = np.empty(self.sel_indices.shape)
            # get cluster numbers only for the selected indices
            pre_select_clusters = self.system.clusters[self.sel_indices]
            for i, _ in enumerate(count):
                probability[pre_select_clusters == i] = prob[i]

            print(probability.shape)
            print(self.sel_indices.shape)

            # norm probability to 1
            probability = probability / probability.sum()
            # select
            self.sel_indices = np.random.choice(
                self.sel_indices, max_s, p=probability, replace=False
            )

        else:
            if not self.system.data["clusters"] is None:
                self.lprint("cannot use fingerprints, clusters.raw file is missing")

            # randomly select indices, the number that is selected is such that
            # taht after deleting these number of selected indices will fit into
            # the max_structures limit
            self.sel_indices = np.random.choice(self.sel_indices, max_s, replace=False)

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

        # because we have masked system some structures might not be available
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

    def n_from_cluster(self, *, n=int):
        self.lprint(f"based on: select {n} random frames from each cluster criterion")

        if not self.system.has_clusters():
            raise RuntimeError(
                "cannot use this criterion, cluster labels are not computed"
            )
        else:
            selected = []
            for label in np.unique(self.system.clusters):
                idx = np.argwhere(self.system.clusters == label).flatten()
                np.random.shuffle(idx)
                selected.append(idx[:n])  # this way it does not fail if n > len(idx)

            self._record_select(np.concatenate(selected))

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
        energies = np.column_stack([p.data["energies"] for p in predictions])
        if per_atom:
            energies /= self.system.get_natoms()

        if std_method:
            e_std = np.std(energies, axis=1)
        else:
            reference = self.system.data["energies"] / self.system.get_natoms()
            # make column vector of reference DFT data
            reference = np.atleast_2d(reference).T

            e_std = np.sqrt(np.mean(np.power(abs(energies - reference), 2), axis=1))

        # set elements that where already selectedin in previous iteration to 0
        e_std[self.system.get_subsystem_indices()] = 0

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

        print(forces.shape)

        # shape: (n_frames, n_atoms, 3)
        if std_method:
            f_std = np.std(forces, axis=0)
        else:
            reference = self.system.data["forces"]
            f_std = np.sqrt(np.mean(np.power(abs(forces - reference), 2), axis=0))

        # shape: (n_frames, n_atoms)
        f_std_size = np.linalg.norm(f_std, axis=2)

        # shape: (n_frames, )
        f_std_max = np.max(f_std_size, axis=1)

        # set elements that where already selectedin in previous iteration to 0
        f_std_max[self.system.get_subsystem_indices()] = 0

        # save for plotting
        self.system.data["forces_std_max"] = f_std_max

        s = self._where((bracket[0] < f_std_max) & (f_std_max < bracket[1]))

        self.lprint(
            f"selected {len(s)} frames as a result of max forces std constraints"
        )

        self._record_select(s)

    def apply(self) -> LabeledSystemMask:

        if self.append:
            diff_frames = len(self.system.get_subsystem_indices()) - len(
                self.sel_indices
            )
            self.lprint(
                f"selecting {len(self.sel_indices)} frames in current iteration"
            )
            if diff_frames > 0:
                self.lprint(
                    f"other {diff_frames} frames will be added as a result of "
                    f"selection from prevous iterations"
                )

        # apply max structures limit if
        # upper limit on number of selected structures is specified
        if self.max_structures is not None:
            self.lprint(f"applying max structures ({self.max_structures}) limit")
            self._limit_structures()

        # append to existing mask
        self.system.append_mask(self.sel_indices)

        # create subsystem filtered by mask
        sub_system = self.system.get_subsystem(None)

        # copy custom attributes
        sub_indices = self.system.get_subsystem_indices()

        for attr in ["energies_std", "forces_std_max"]:
            try:
                sub_system.data[attr] = self.system.data[attr][sub_indices]
            except KeyError:
                print(f"Cannot copy key: {attr} to subsystem")

        return sub_system
