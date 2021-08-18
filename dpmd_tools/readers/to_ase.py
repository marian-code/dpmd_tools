from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Optional
from .common import get_lmp_traj_indices
from ase.io.lammpsrun import read_lammps_dump_text
from joblib import Parallel, delayed
from tqdm import tqdm
import threading

from dpdata import LabeledSystem

if TYPE_CHECKING:
    from ase import Atoms


def load_npy_data(path: Path) -> List["Atoms"]:

    system = LabeledSystem()

    system.from_deepmd_comp(str(path.resolve()))
    return system.to_ase_structure()


def load_raw_data(path: Path) -> List["Atoms"]:
    return LabeledSystem(str(path), fmt="deepmd/raw").to_ase_structure()


def load_dev_lmp_traj(
    traj_file: Path,
    dev_energy: Tuple[float, float],
    dev_force: Tuple[float, float],
    portion: Optional[float],
    nframes: Optional[int],
    lammps_infile: str = "in.lammps",
    plummed_infile: str = "plumed.dat",
) -> List["Atoms"]:
    from py_extract_frames.extract_frames import copy_traj, parse_traj

    indices, _ = get_lmp_traj_indices(
        traj_file.parent, dev_energy, dev_force, lammps_infile, plummed_infile
    )

    print(f"got {len(indices)} frames to recompute")

    # should be sorted but better be sure
    indices.sort()

    if portion:
        if portion < 0 or portion > 1:
            raise ValueError("Portion must be between 0 and 1") 
        nframes = len(indices) * portion
    elif not portion and not nframes:
        raise RuntimeError("must specify 'portion' or 'nframes'")

    # take every n-th frame as specified by portion
    indices = indices[::int(len(indices) / nframes)]

    print(f"selected {len(indices)} frames to recompute")

    atoms = []

    def get_frame(index: int) -> "Atoms":
        traj_frame = Path(f"/tmp/traj_{threading.get_ident()}.tmp")

        copy_traj(traj_file, traj_frame, *parse_traj(traj_file, index, index))
        
        with traj_frame.open("r") as f:
            return read_lammps_dump_text(f)

    # extract one frame into file and than read to ASE one by one
    #for i in tqdm(indices):
        #atoms.append(get_frame(i))
    
    pool = Parallel(n_jobs=2, backend="threading")
    exec = delayed(get_frame)
    atoms = pool(exec(i) for i in tqdm(indices))

    # remove last tmp trajectory files
    for f in Path("/tmp").glob("traj_*.tmp"):
        f.unlink()

    print("Saving to xyz format")

    return atoms

