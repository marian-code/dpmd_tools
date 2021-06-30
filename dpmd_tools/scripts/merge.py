from os import mkdir
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def main():

    data_dirs = [d for d in Path.cwd().rglob("*") if (d / "type.raw").is_file()]
    n_atoms = [len((d / "type.raw").read_text().splitlines()) for d in data_dirs]

    groups: Dict[int, List[Path]] = defaultdict(list)
    for d, n in zip(data_dirs, n_atoms):
        groups[n].append(d)

    for key, value in groups.items():
        print(key)
        for v in value:
            print(f"    - {v}")

    FILES = ("box", "coord", "energy", "force", "virial")

    for group, dirs in groups.items():

        print(f"merging {len(dirs):2} dataset(s) in {group} ...", end="")
        
        # check if files with type info match
        assert len(set([(d / "type.raw").read_text() for d in dirs])) == 1, "The types.raw files for all dirs are not equal"
        assert len(set([(d / "type_map.raw").read_text() for d in dirs])) == 1, "The type_map.raw files for all dirs are not equal"

        # check if all raw files are present
        for f in FILES:
            df = f"{f}.raw"  # data file
            assert all([(d / f"{df}").is_file() for d in dirs]), f"data file {df} is not present in all dirs"

        # create merge directory
        target = Path.cwd() / f"Ge{group}"
        target.mkdir(exist_ok=True)

        # merge files 
        for f in FILES:
            df = f"{f}.raw"  # data file
            with (target / df).open("w") as merge_f:
                for d in dirs:
                    with (d / df).open("r") as f:
                        shutil.copyfileobj(f, merge_f)

        print(" done")


if __name__ == "__main__":
    main()