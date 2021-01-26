"""Helper modulre with dpdata subclasses."""

from pathlib import Path
from typing import ItemsView, Iterator, KeysView, List, Union, ValuesView
from typing_extensions import Literal
import numpy as np
from tqdm import tqdm
from dpdata import LabeledSystem, MultiSystems

MAX_FRAMES_PER_SET = 5000


class MultiSystemsVar(MultiSystems):

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
            The size of each set.
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

    def from_xtalopt(base_dir: Path):
        ...
    def from_outcars(files: List[Path]):
        ...
    def from_dp_raw():
        ...


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

