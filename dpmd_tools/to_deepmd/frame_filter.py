"""Houses class that takes care of frame filtering for datasets construction."""

import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, NoReturn, Tuple, Union

import numpy as np
from ase import units
from colorama import Fore, init
from dpdata import LabeledSystem
from dpmd_tools.system import ClusteredSystem, MaskedSystem

init(autoreset=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ["KMP_WARNINGS"] = "FALSE"

WORK_DIR = Path.cwd()


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
        system: Union[MaskedSystem, ClusteredSystem],
        use_prints: bool,
        lprint: Callable[[Any], NoReturn],
        append: bool = True,
        cache_predictions: bool = False,
    ) -> None:

        self.lprint = lprint
        self.system = system
        self.system_mutated = False
        self.cache_predictions = cache_predictions
        self.use_prints = use_prints
        self.sel_indices = np.arange(self.system.get_nframes(), dtype=int)
        self.append = append
        self._init()

    def _init(self):
        self._selection_start(f"previous {self.system.iteration} iterations")
        self.lprint(f" - selected {len(self.system.get_subsystem_indices())} frame(s)")

    def get_predictions(self, graphs: List[str]) -> List[MaskedSystem]:

        if not self._predictions:
            self.lprint(f" - computing model predictions")
            self._predictions = []
            for i, g in enumerate(graphs, 1):
                self.lprint(f" - graph {i}/{len(graphs)}")

                # load/save cached predictions to avoid recomputing when we need to run
                # again. This happens quite often since we need to guess bracket
                # intervals and there is not knowing beforehand how many structures this
                # will yield
                cache_dir = WORK_DIR / f"cache_{g.name}"
                if self.cache_predictions:
                    if cache_dir.is_dir():
                        self.lprint(
                            f"    - {Fore.LIGHTGREEN_EX}found cached predictions: "
                            f"{Fore.RESET}{cache_dir.relative_to(WORK_DIR)}"
                        )
                        system = LabeledSystem(cache_dir, fmt="deepmd/npy")
                    else:
                        self.lprint(
                            f"    - {Fore.YELLOW}could not find cached predictions: "
                            f"{Fore.RESET}{cache_dir.relative_to(WORK_DIR)}"
                        )
                        system = self.system.predict(g)
                else:
                    system = self.system.predict(g)

                self._predictions.append(system)

                if not cache_dir.is_dir():
                    self.lprint(
                        f"    - saving predictions to {cache_dir.relative_to(WORK_DIR)}"
                    )
                    system.to_deepmd_npy(cache_dir)
        return self._predictions

    def _record_select(self, sel_indices: np.ndarray):
        self.sel_indices = np.intersect1d(self.sel_indices, sel_indices)
        self.lprint(f" - selected {len(sel_indices)} frame(s)")

    def _where(self, cond) -> np.ndarray:
        return np.argwhere(cond & (self.system.mask == 0)).flatten()

    def _selection_start(self, msg: str):
        self.lprint(f"{Fore.LIGHTBLUE_EX}based on {msg}")

    def max_select(self, max_n: str):

        self._selection_start(f"max structures ({max_n}) limit")

        if "%" in max_n:
            max_s = int(
                self.system.get_nframes() * (float(max_n.replace("%", "")) / 100)
            )
            self.lprint(f" - {max_n} portion equals {max_s} frames")
        else:
            max_s = int(max_n)

        over_limit = len(self.sel_indices) - max_s

        # if the number of selected structures is not over limit, return
        if over_limit <= 0:
            self.lprint(
                f" - number of selected frames {len(self.sel_indices)} is lower "
                f"than set max frames limit, skipping this criterion ..."
            )
            return

        if self.use_prints and self.system.has_clusters:
            self.lprint(
                f" - {Fore.LIGHTGREEN_EX}found {Fore.RESET}clusters.raw"
                f"{Fore.LIGHTGREEN_EX} file and will use fingerprints based clustering "
                f"to select the most diverse structures"
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

            # norm probability to 1
            probability = probability / probability.sum()
            # select
            s = np.random.choice(self.sel_indices, max_s, p=probability, replace=False)

        else:
            if not self.system.has_clusters:
                self.lprint(
                    f" - {Fore.YELLOW}cannot use fingerprints, {Fore.RESET}clusters.raw "
                    f"{Fore.YELLOW}file is missing"
                )

            # randomly select indices, the number that is selected is such that
            # taht after deleting these number of selected indices will fit into
            # the max_structures limit
            s = np.random.choice(self.sel_indices, max_s, replace=False)

        self._record_select(s)

    @check_bracket
    def energy(self, *, bracket: Tuple[float, float], per_atom: bool = True):
        self._selection_start(f"energy{' per atom' if per_atom else ''}")

        energies = self.system.data["energies"]
        if per_atom:
            energies = energies.copy() / self.system.get_natoms()

        s = self._where((bracket[0] < energies) & (energies < bracket[1]))

        self._record_select(s)

    def every(self, *, n_th: int):
        self._selection_start(f"take every {n_th} frame")

        indices = np.arange(self.system.get_nframes())

        # because we have masked system some structures might not be available
        # so we try to shift the choose indices by one in each loop and choose
        # the result that gives most structures
        s = []
        for i in range(n_th):
            s.append(self._where((indices % n_th == i)))

        s = sorted(s, key=lambda x: len(x), reverse=True)[0]

        self._record_select(s)

    def n_from_cluster(self, *, n=int):
        self._selection_start(f"select {n} random frames from each cluster")

        if not self.system.has_clusters:
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
    def pressure(self, *, bracket: Tuple[float, float]):
        """The pressure does not exactly correspond to one output by VASP.

        This is a result of dpdata applying affine transormation.
        """
        self._selection_start(f"pressure")

        v_pref = units.GPa / units.eV

        volumes = np.linalg.det(self.system.data["cells"])
        pressure_Gpa = np.trace(
            self.system["virials"] / v_pref / volumes[:, None, None], axis1=1, axis2=2
        )
        print(pressure_Gpa)

        s = self._where((bracket[0] < pressure_Gpa) & (pressure_Gpa < bracket[1]))

        self._record_select(s)

    @check_bracket
    def volume(self, *, bracket: Tuple[float, float], per_atom: bool = True):
        self._selection_start(f"volume{' per atom' if per_atom else ''}")

        # this an array of square matrices, linalg det auto calculates det for
        # every sub matrix that is square
        volumes = np.linalg.det(self.system.data["cells"])
        if per_atom:
            volumes /= self.system.get_natoms()

        s = self._where((bracket[0] < volumes) & (volumes < bracket[1]))

        self._record_select(s)

    @check_bracket
    def dev_e(
        self,
        *,
        graphs: List[str],
        bracket: Tuple[float, float],
        std_method: bool = False,
        per_atom: bool = True,
        from_md: bool = False,
    ):
        """Select which labeled structures should be added to dataset.

        This method is usefull when you already have DFT data labels without
        any prior selection and want to decide which of them should be added to
        the dataset based on dataset predictions for them.

        See Also
        --------
        :func:`dpgen.simplify.simplify.post_model_devi`
        """
        self._selection_start(f"energy std")

        if not from_md:
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
            self.system = self.system.add_dev_e(e_std)
            self.system_mutated = True

        s = self._where((bracket[0] < e_std) & (e_std < bracket[1]))

        self._record_select(s)

    @check_bracket
    def dev_f(
        self,
        *,
        graphs: List[str],
        bracket: Tuple[float, float],
        std_method: bool = False,
        from_md: bool = False,
    ):
        """Select which labeled structures should be added to dataset.

        This method is usefull when you already have DFT data labels without
        any prior selection and want to decide which of them should be added to
        the dataset based on dataset predictions for them.

        See Also
        --------
        :func:`dpgen.simplify.simplify.post_model_devi`
        """
        self._selection_start(f"max atom force std")

        if not from_md:
            predictions = self.get_predictions(graphs)

            # shape: (n_models, n_frames, n_atoms, 3)
            forces = np.stack([p.data["forces"] for p in predictions])

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
            self.system = self.system.add_dev_f(f_std_max)
            self.system_mutated = True

        s = self._where((bracket[0] < f_std_max) & (f_std_max < bracket[1]))

        self._record_select(s)

    def apply(self) -> MaskedSystem:

        self.lprint(f"{Fore.LIGHTBLUE_EX}applying filters from all conditions")

        if self.append:
            self.lprint(
                f" - selecting {len(self.sel_indices)} frames in current iteration"
            )
            if len(self.system.get_subsystem_indices()) > 0:
                self.lprint(
                    f" - other {len(self.system.get_subsystem_indices())} frames "
                    f"will be added as a result of selection from prevous iterations"
                )

        # append to existing mask
        self.system.append_mask(self.sel_indices)

        # create subsystem filtered by mask
        sub_system = self.system.get_subsystem(None)

        return sub_system
