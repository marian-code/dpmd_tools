"""Helper modulre with dpdata subclasses."""

import concurrent.futures as cf
import math
from pathlib import Path
from typing import (TYPE_CHECKING, Callable, ItemsView, Iterator, KeysView,
                    List, Union, ValuesView)

import numpy as np
from dpdata import LabeledSystem, MultiSystems
from tqdm import tqdm
from typing_extensions import Literal

if TYPE_CHECKING:
    from ase import Atoms

MAX_FRAMES_PER_SET = 5000


class MultiSystemsVar(MultiSystems):
    """Conveniens subclass of Multisystems.

    Adds `pathlib.Path` compatibility and few other convenience methods.
    Also introduces better auto set partitioning.
    """

    def to_deepmd_npy(self, folder: Path,
                      set_size: Union[int, list, Literal["auto"]] = 5000,
                      prec=np.float32):
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
            assert len(set_size) == self.__len__(), ("Incorrect length of "
                                                     "set_size")

        dump_job = tqdm(zip(set_size, self.systems.items()), ncols=100,
                        total=len(set_size))

        for ss, (system_name, system) in dump_job:
            system.to_deepmd_npy(str((folder / system_name).resolve()),
                                 set_size=ss, prec=prec)

    def to_deepmd_raw(self, folder: Path):
        super().to_deepmd_raw(str(folder.resolve()))

    def keys(self) -> KeysView[str]:
        return self.systems.keys()

    def items(self) -> ItemsView[str, LabeledSystem]:
        return self.systems.items()

    def values(self) -> ValuesView[LabeledSystem]:
        return self.systems.values()

    def shuffle(self):
        for system in self.values():
            system.shuffle()

    def predict(self, graphs: List[Path]) -> Iterator["MultiSystemsVar"]:

        for g in graphs:
            prediction = super().predict(str(g.resolve()))
            multi_sys = MultiSystemsVar()
            
            # copy data
            for attr in vars(prediction):
                setattr(multi_sys, attr, getattr(prediction, attr))

            yield multi_sys

    def collect_single(self, paths: List[Path],
                       dir_process: Callable[[Path], List[LabeledSystem]]):
        """Single core serial data collector"""

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

    def collect_cf(self, paths: List[Path],
                   dir_process: Callable[[Path], List[LabeledSystem]]):
        """Parallel async data collector."""

        with cf.ProcessPoolExecutor(max_workers=10) as pool:
            future2data = {pool.submit(dir_process, p): p for p in paths}

            futures = tqdm(cf.as_completed(future2data),
                        ncols=100, total=len(paths))
            for future in futures:
                path = future2data[future].name
                futures.set_description(f"extracting: {path}")
                try:
                    systems = future.result()
                except Exception as e:
                    futures.write(f"Error in {path}: {e}")
                else:
                    for s in systems:
                        try:
                            s.data["atom_names"] = ["Ge"]
                            self.append(s)
                        except Exception as e:
                            futures.write(f"Error in {path}: {e}")


def split_into(n: int, p: int):
    """Split number n to p parts as evenly as possible.

    Example: split_into(32, 3) => [11, 11, 10]
    """
    split = [n / p + 1] * (n % p) + [n / p] * (p - n % p)
    return [int(s) for s in split]


def partition_systems(multi_sys: MultiSystemsVar) -> List[int]:

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
                print(f"system {name} with {sys_frames} frames, will "
                      f"be partitioned to sets counting {set_frames}")
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


def load_npy_data(path: Path) -> List["Atoms"]:

    system = LabeledSystem()

    system.from_deepmd_comp(str(path.resolve()))
    return system.to_ase_structure()


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])
