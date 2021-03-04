"""Helper module with dpdata subclasses."""

from concurrent.futures import as_completed
from pathlib import Path
from typing import (
    Callable,
    Generic,
    ItemsView,
    Iterator,
    KeysView,
    List,
    TypeVar,
    Union,
    ValuesView,
)
from warnings import warn

import numpy as np
from colorama import Fore
from dpdata import LabeledSystem, MultiSystems
from dpmd_tools.utils import split_into
from loky import get_reusable_executor
from tqdm import tqdm
from typing_extensions import Literal

from .flavours import AllSelectedError, ClusteredSystem, MaskedSystem, SelectedSystem

MAX_FRAMES_PER_SET = 5000
_SYS_TYPE = TypeVar("_SYS_TYPE", MaskedSystem, ClusteredSystem, SelectedSystem)


class MultiSystemsVar(MultiSystems, Generic[_SYS_TYPE]):
    """Conveniens subclass of Multisystems.

    Adds `pathlib.Path` compatibility and few other convenience methods.
    Also introduces better auto set partitioning.
    """

    def to_deepmd_npy(
        self,
        folder: Path,
        set_size: Union[int, list, Literal["auto"]] = 5000,
        prec=np.float32,
    ):
        """Dump the system in deepmd compressed format.

        Dump numpy binary (.npy) to `folder` for each system.

        Parameters
        ----------
        folder : str
            The output folder
        set_size : int
            The size of each set, if `auto` than the system will be partitioned
            to as equally long sets as possible while also trying to maintain
            data split as close to 90:10 and strictly obeying condition for
            MAX_FRAMES_PER_SET
        prec: {numpy.float32, numpy.float64}
            The floating point precision of the compressed data
        """
        if isinstance(set_size, int):
            set_size = [set_size for _ in range(self.__len__())]
        elif set_size == "auto":
            set_size = partition_systems(self)
        else:
            assert len(set_size) == self.__len__(), "Incorrect length of set_size"

        dump_job = tqdm(
            zip(set_size, self.systems.items()), ncols=100, total=len(set_size)
        )

        for ss, (system_name, system) in dump_job:
            system.to_deepmd_npy(folder / system_name, set_size=ss, prec=prec)

    # TODO something fishy is going on here
    # npy and raw files do not correspond
    def to_deepmd_raw(self, folder: Path, append: bool):
        for system_name, system in self.systems.items():
            system.to_deepmd_raw(folder / system_name, append)

    def keys(self) -> KeysView[str]:
        return self.systems.keys()

    def items(self) -> ItemsView[str, _SYS_TYPE]:
        return self.systems.items()

    def values(self) -> ValuesView[_SYS_TYPE]:
        return self.systems.values()

    def shuffle(self):
        for system in self.values():
            system.shuffle()

    @property
    def iteration(self) -> int:
        it = set([s.iteration for s in self.values()])
        if len(it) > 1:
            warn(
                f"not all subsystems have the same number of selection iterations: {it}"
                f"this concerns only graph export names, so mostly you can safely "
                f"ignore this message"
            )

        return max(it)

    def predict(self, graphs: List[Path]) -> Iterator["MultiSystemsVar"]:

        for g in graphs:
            prediction = super().predict(str(g.resolve()))
            multi_sys = MultiSystemsVar()

            # copy data
            for attr in vars(prediction):
                setattr(multi_sys, attr, getattr(prediction, attr))

            yield multi_sys

    def collect_debug(
        self, paths: List[Path], dir_process: Callable[[Path], List[_SYS_TYPE]]
    ):
        """Single core serial data collector with no exception catching.
        
        Use only for debugging.
        """
        print(f"{Fore.RED} using debugging file reader")

        for path in paths:
            systems = dir_process(path)
            for s in systems:
                self.append(s)

    def collect_single(
        self, paths: List[Path], dir_process: Callable[[Path], List[_SYS_TYPE]]
    ):
        """Single core serial data collector."""
        futures = tqdm(paths, ncols=100, total=len(paths))

        for p in futures:

            try:
                systems = dir_process(p)
            except Exception as e:
                futures.write(f"Error in {p.name}: {e}")
            else:
                for s in systems:
                    try:
                        self.append(s)
                    except Exception as e:
                        futures.write(f"Error in {p.name}: {e}")

    def collect_cf(
        self, paths: List[Path], dir_process: Callable[[Path], List[_SYS_TYPE]],
    ):
        """Parallel async data collector."""
        print("All atom types will be changed to Ge!!!")

        with get_reusable_executor(max_workers=10) as pool:
            future2data = {pool.submit(dir_process, p): p for p in paths}

            futures = tqdm(as_completed(future2data), ncols=100, total=len(paths))
            for future in futures:
                path = future2data[future].name
                futures.set_description(f"extracting: {path}")
                try:
                    systems = future.result()
                except AllSelectedError as e:
                    futures.write(
                        f"All structures from system {path} have already been selected,"
                        f" skipping..."
                    )
                except Exception as e:
                    futures.write(f"Error in {path}: {e}")
                else:
                    for s in systems:
                        try:
                            s.data["atom_names"] = ["Ge"]
                            self.append(s)
                        except Exception as e:
                            futures.write(f"Error in {path}: {e}")


def partition_systems(multi_sys: MultiSystemsVar) -> List[int]:
    """Inteligently partition systems to sets.

    Instead of 10, 10, 7 that is output by dpdata -> 9, 9, 9
    Try to preserve 10% test data ratio with max set size == 5000.

    Parameters
    ----------
    multi_sys : MultiSystemsVar
        dpdata mustisystem class

    Returns
    -------
    List[int]
        list with partitioning for each system

    Raises
    ------
    RuntimeError
        If partitioning fails (too many structures)
    """
    d: LabeledSystem
    n_frames = []
    n_tests = []
    for name, d in multi_sys.items():
        d.shuffle()
        sys_frames = d.get_nframes()

        # ideally split to 10% portions the last will be used for testing
        # if 10% produces too big set size, lower that portion until the
        # condition is satified
        for parts in range(10, 100):
            sets = split_into(sys_frames, parts)

            set_frames = sets.pop(0)
            set_tests = sets.pop(len(sets) - 1)

            # the difference when e.g. split will be [30, 30, 29, 29]
            # dpdata will split to: [30, 30, 30, 28]
            # diff accounts for that
            diff = 0
            for i in range(len(sets)):
                diff += sets[0] - sets[i]

            set_tests -= diff

            if set_frames < MAX_FRAMES_PER_SET:
                print(
                    f"system {name} with {sys_frames} frames, will "
                    f"be partitioned to sets counting {set_frames}"
                )
                n_frames.append(set_frames)
                n_tests.append(set_tests)
                break
        else:
            raise RuntimeError(
                f"Couldn't split system {name} with {sys_frames} to parts "
                f"while satisfying the condition to keep set size under: "
                f"{MAX_FRAMES_PER_SET}"
            )

    return n_frames
