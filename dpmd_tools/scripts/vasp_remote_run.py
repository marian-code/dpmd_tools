from io import TextIOBase
import sys
from typing import Any, Dict

from ssh_utilities import Connection


class ProxyStream:

    def __init__(self, stream: TextIOBase, filename: str) -> None:
        self.file = open(filename, "w")
        self.stream = stream

    def write(self, text: str):

        self.stream.write(text)
        self.file.write(text)

    def close(self):
        self.file.close()


def run_vasp(args: Dict[str, Any]):

    with Connection(args["server"], local=False, quiet=True) as c:

        c.os.makedirs(args["remote"], exist_ok=True, parents=True)

        print("uploading job files")
        FILES = ("INCAR", "KPOINTS", "POSCAR", "POTCAR")
        for f in FILES:
            c.shutil.copy2(
                args["local"] / f,
                args["remote"] / f,
                direction="put",
                follow_symlinks=False,
            )

        if args["server"] == "fock":
            VASP = "/home/s/Software/vasp.5.4.4_mpi_TS/bin/vasp_std"
        else:
            VASP = "/home/s/Software/VASP/intel-mpi-TS-HI/vasp-5.4.4-TS-HI/bin/vasp_std"

        print("running VASP...")
        print("***********************************************************************")
        c.subprocess.run(
            [
                "source",
                "/opt/intel/bin/compilervars.sh",
                "-arch",
                "intel64",
                "&&",
                "/home/s/bin/mpirun",
                "-np",
                "12",
                VASP,
            ],
            suppress_out=True,
            quiet=True,
            stdout=ProxyStream(sys.stdout, "output.txt"),
            stderr=ProxyStream(sys.stderr, "error.txt"),
            cwd=args["remote"],
            encoding="utf-8",
        )
        print("***********************************************************************")
        print("done")

        print("downloading results")
        c.shutil.download_tree(
            args["remote"], args["local"], exclude="*WAVECAR", quiet=True
        )
