from pathlib import Path
from typing import List, Optional, Tuple
from ssh_utilities import Connection
import shutil
from atexit import register

#logging.basicConfig(level=logging.DEBUG)
from ssh_utilities.remote.path import SSHPath

WORK_DIR = Path.cwd()


def upload(args: dict):

    # expand wildcards
    dirs = []
    for d in args["dirs"]:
        dirs.extend([f for f in WORK_DIR.glob(d)])

    # get server and path
    server = args["server"]
    if args["target"]:
        target = Path(args["target"])
    else:
        target = WORK_DIR

    # init cconnection
    c = Connection.get(args["server"], local=False, quiet=True)
    register(c.close, quiet=True)

    if server:
        target_dir = c.pathlib.Path(target)
        target_dir.mkdir(exist_ok=True, parents=True)

    if args["local"]:
        local_dir = Path(args["local"]).resolve()
        local_dir.mkdir(exist_ok=True, parents=True)

    # map sources and targets
    dirs_mapping: List[Tuple[Path, Optional[SSHPath], Optional[Path]]] = []
    for d in dirs:
        data = d / "deepmd_data/for_train"
        if data.is_dir():
            if args["local"] and server:
                entry = (data, target_dir / d.name, local_dir / d.name)
            elif args["local"]:
                entry = (data, None, local_dir / d.name)
            elif server:
                entry = (data, target_dir / d.name, None)
            dirs_mapping.append(entry)
        else:
            print(f"could not find data in {data}")

    print("upload summary:")
    print("------------------------------------------------------------------")
    for src, dst, loc in dirs_mapping:
        print(f"{src.relative_to(Path.cwd())} -> {server}@{dst} -> {loc}")

    carry_on = input("continue? [ENTER]")
    if carry_on != "":
        assert False, "Aborting"

    print("\nuploading ...")
    print("------------------------------------------------------------------")
    for src, dst, local_dst in dirs_mapping:
        if dst:
            dst.mkdir(exist_ok=True, parents=True)
            c.shutil.upload_tree(src, dst, remove_after=False, quiet="stats")
        if local_dst:
            try:
                print(f"\ncopying to local: {local_dst.name}... ", end="")
                shutil.copytree(src, local_dst, dirs_exist_ok=True)
            except FileExistsError as e:
                print(e)
            else:
                print("OK")

    print("\ntraining dirs")
    print("------------------------------------------------------------------")
    data_dirs = []
    for _, dst, loc in dirs_mapping:
        if dst:
            dirs = [d for d in dst.glob("*") if (d / "box.raw").is_file()]
        elif loc:
            dirs = [d for d in loc.glob("*") if (d / "box.raw").is_file()]

        d = [d.relative_to(target_dir) for d in dirs]
        data_dirs.extend(d)

    for d in data_dirs:
        print(f"- {d}")
