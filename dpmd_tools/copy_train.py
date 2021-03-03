#!/opt/miniconda3/bin/python
# PYTHON_ARGCOMPLETE_OK
import argparse
from typing import Dict, Sequence, Union
import argcomplete
from pathlib import Path
import yaml
import json
import shutil
import re
import random

WORK_DIR = Path.cwd()


def suggest_control(prefix: str, parsed_args, **kwargs) -> Sequence[str]:
    if parsed_args.input:
        path = Path(parsed_args.input)
        files = [f for f in path.glob("*.json")]
        files.extend([f for f in path.glob("*.yaml")])
        files.extend([f for f in path.glob("*.yml")])
        try:
            str_files = [str(f.relative_to(parsed_args.input)) for f in files]
        except ValueError:
            str_files = [str(f) for f in files]
        if prefix:
            str_files = [f for f in str_files if prefix in f]
        return str_files
    else:
        argcomplete.warn("specify -i/--input first!")
        return []


def input_parser() -> Dict[str, Union[str, Path]]:

    p = argparse.ArgumentParser(
        description="make new training direstory from existing one",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="choose directory from which will new train dir be created",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="choose destination directory",
    )
    p.add_argument(
        "-c",
        "--control",
        type=str,
        required=True,
        help="choose training control file",
    ).completer = suggest_control

    argcomplete.autocomplete(p)

    return vars(p.parse_args())


def prepare(*, input: Path, output: Path, control: str):

    # copy directory
    shutil.copytree(input, output)

    # delete unnecessary files
    delete_files = set()

    with (output / control).open() as fp:
        if control.endswith("json"):
            data = json.load(fp)["training"]
        elif control.endswith(("yml", "yaml")):
            data = yaml.safe_load(fp)["training"]

        delete_files.add(data["disp_file"])
        for f in output.glob(f"{data['save_ckpt']}*"):
            delete_files.add(f.name)

    delete_files.add("checkpoint")
    for f in output.glob("*.txt"):
        delete_files.add(f.name)

    delete_files.add("out.json")
    for f in output.glob("*.pb"):
        delete_files.add(f.name)

    for d in delete_files:
        try:
            (output / d).unlink()
        except FileNotFoundError:
            pass

    train_num = int(re.findall(r"\S*?(\d+)", output.name)[0])

    for f in output.glob("*"):
        text = f.read_text()

        #  control files
        if "seed" in text:
            # for yaml
            text = re.sub(
                r"seed\s*:\s*\d+", f"seed: {random.randint(1, 10**10)}", text
            )
            # for json
            text = re.sub(
                r"\"seed\"\s*:\s*\d+", f'"seed": {random.randint(1, 10**10)}',
                text
            )
        # for pbs files
        if "PBS" in text or "SBATCH" in text:
            text = re.sub(r"train_\d+", f"train_{train_num}", text)
            text = re.sub(
                r"((?:-o|--output)\s*\S*?)\d+.pb", r"\g<1>{}.pb".format(train_num),
                text
            )

        f.write_text(text)


if __name__ == "__main__":
    args = input_parser()
    prepare(**args)