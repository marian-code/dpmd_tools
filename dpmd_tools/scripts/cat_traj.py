import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union, overload

from tqdm import tqdm
from typing_extensions import Literal

WORK_DIR = Path.cwd()


def input_parser():
    p = argparse.ArgumentParser(
        description="Join lammps trajectories from more files. This script "
        "will also modify trajectory files in place so there are no "
        "overlaping steps in sucessive trajectory files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("-t", "--trajectories", nargs="+", type=Path, default=[],
                   help="enter names of trajectory files you wish to append, "
                   "order will be determined automatically, if argument is "
                   "not used files will be detected automatically")
    p.add_argument("-o", "--outfile", default="all_traj.lammps", type=str,
                   help="name of the output merged trajectory file")

    args = p.parse_args()

    return [Path(t) for t in args.trajectories], args.outfile


def sort_trajectories(files: List[Path]) -> List[Path]:
    """Sort trajectory files according to their first step."""
    return sorted(files, key=lambda x: find_first_step(x))


def iter_shifted_pair(iterator: Iterator) -> Iterator:
    """Iterate through sucessive element pairs of the original iterator."""
    iter_normal, iter_shift = itertools.tee(iterator)
    next(iter_shift)
    return zip(iter_normal, iter_shift)


def find_dump_freq(path: Path) -> int:
    """Find dump frequency from trajectory file."""
    with path.open("r") as f:
        steps = []
        for line, next_line in iter_shifted_pair(f):
            if "ITEM: TIMESTEP" in line:
                steps.append(int(next_line))
            if len(steps) == 2:
                return steps[1] - steps[0]


@overload
def read_reverse(path: Path, yield_pointer: Literal[False]) -> Iterator[str]:
    ...

@overload
def read_reverse(path: Path, yield_pointer: Literal[True]
                 ) -> Iterator[Tuple[int, str]]:
    ...

def read_reverse(path: Path, yield_pointer: Literal[True, False] = False
                 ) -> Union[Iterator[str], Iterator[Tuple[int, str]]]:
    """Iterate file lines in reverse order, without reading whole file."""
    # Open file for reading in binary mode
    with path.open('rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character
            # then it means one line is read
            if new_byte == b'\n':
                # Fetch the line from buffer and yield it
                if yield_pointer:
                    yield pointer_location, buffer.decode()[::-1]
                else:
                    yield buffer.decode()[::-1]
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)

        # As file is read completely, if there is still data in buffer,
        # then its the first line.
        if len(buffer) > 0:
            # Yield the first line too
            if yield_pointer:
                yield pointer_location, buffer.decode()[::-1]
            else:
                yield buffer.decode()[::-1]


def find_first_step(path: Path) -> int:
    """Find the very first timestep value."""
    with path.open("r") as f:
        for line, next_line in iter_shifted_pair(f):
            if "ITEM: TIMESTEP" in line:
                return int(next_line)


def find_last_step(path: Path) -> int:
    """Find the last timestep value."""
    f = read_reverse(path, yield_pointer=False)
    for next_line, line in iter_shifted_pair(f):
        if "ITEM: TIMESTEP" in line:
            return int(next_line)


def check_continuity(files: List[Path]) -> bool:
    """Check if trajectory chunks contain continuous steps."""
    continuous = True

    test_data = []
    for f in files:
        test_data.append({
            "dump_freq": find_dump_freq(f),
            "start": find_first_step(f),
            "stop": find_last_step(f),
            "name": f.name
        })

    print("checking if dump frequency of all trajectories matches", end="")
    dump_freq = set([t["dump_freq"] for t in test_data])

    if len(dump_freq) == 1:
        print(" ... OK")
    else:
        print(" ... ERROR")
        print(f"Found {len(dump_freq)} different dump frequencies {dump_freq}")
        continuous = False

    print("checking if timesteps in all files are continuous")

    for t, t_next in iter_shifted_pair(test_data):
        print(f"checking {t['name']}", end="")
        if t_next["start"] <= t["stop"]:
            print(" ... OK")
        elif t_next["start"] == t["stop"] + t["dump_freq"]:
            print(" ... OK")
        else:
            print(" ... ERROR")
            print(f"There is a gap in the trajectory data {t['name']} last "
                  f"step is: {t['stop']} and next trajectory file "
                  f"{t_next['start']} first step is {t_next['stop']} with "
                  f"dump frequency {t['dump_freq']}")
            continuous = False

    return continuous


def find_line_w_step(traj_file: Path, find_step: int) -> Optional[int]:
    """Find byte pointer of line that contains the specified step."""
    line_pair = iter_shifted_pair(read_reverse(traj_file, yield_pointer=True))

    for (_, next_line), (pointer, line) in line_pair:
        if "ITEM: TIMESTEP" in line:
            next_step = int(next_line)
            if next_step < find_step:
                return None
            if next_step == find_step:
                return pointer

    return None


def trim_files(traj_files: List[Path]):
    """Trim sucessive trajectory files so they don't contain same steps."""

    for traj, traj_next in iter_shifted_pair(traj_files):

        first_step = find_first_step(traj_next)
        pointer = find_line_w_step(traj, first_step)

        if pointer:
            head(traj, pointer)


def head(traj_file: Path, pointer: int):
    """Take head of file until position defined by byte pointer."""
    with traj_file.open("r+") as f:
        f.truncate(pointer + 1)


def cat_files(traj_files: List[Path], traj_outfile: Path):
    """Catenate files together to one big file."""
    with traj_outfile.open("w") as outfile:

        total_size = sum([f.stat().st_size for f in traj_files])
        shift = 0
        with tqdm(total=total_size, ncols=150, unit="bytes") as pbar:
            for i, tf in enumerate(traj_files, 1):
                pbar.set_description(f"processing file {i}/{len(traj_files)}")
                with tf.open("r") as infile:
                    while True:
                        buf = infile.read(1024 * 64)
                        if not buf:
                            break
                        outfile.write(buf)
                        pbar.update(infile.tell() + shift - pbar.n)

                shift += tf.stat().st_size
                outfile.write("\n")

        print("saving to final file")


def main(traj_files: List[Path], traj_outfile: str):

    traj_files = sort_trajectories(traj_files)

    print("found trajectory files order:")
    for i, tf in enumerate(traj_files):
        print(f"{i:>2d}: {tf}")

    if not check_continuity(traj_files):
        while True:
            inpt = input("Trajectory files are not continuous, do you want "
                         "to merge anyway? [y]/n: ")
            if inpt.lower() in ("", "y", "yes"):
                break
            elif inpt.lower() in ("n", "no"):
                sys.exit()
            else:
                print("input only 'y', 'yes' or 'n', 'no'!")

    trim_files(traj_files)
    cat_files(traj_files, WORK_DIR / traj_outfile)


if __name__ == "__main__":

    traj_files, outfile = input_parser()

    if not traj_files:
        traj_files = [f for f in (WORK_DIR).glob(f"traj*.lammps*")]

    main(traj_files, outfile)
