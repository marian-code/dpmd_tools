from io import TextIOBase
from pathlib import Path
import sys
from colorama import Fore

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


def run_singlepoint(*, server: str, local: Path, remote: Path, what: str, command: str = None):

    if server == "fock":
        VASP = "/home/s/Software/vasp.5.4.4_mpi_TS/bin/vasp_std"
    else:
        VASP = "/home/s/Software/VASP/intel-mpi-TS-HI/vasp-5.4.4-TS-HI/bin/vasp_std"

    if server in ("kohn", "planck"):
        CORES = "16"
    else:
        CORES = "12"

    QE = "/home/toth/Software/qe.6.3.0/bin/pw.x < relax.in"

    with Connection(server, local=False, quiet=True) as c:

        c.os.makedirs(remote, exist_ok=True, parents=True)

        print("uploading job files")
        if what == "QE":
            code = QE
            c.shutil.copy2(
                local / "relax.in",
                remote / "relax.in",
                direction="put",
                follow_symlinks=False,
            )
        else:
            code = VASP

            FILES = ("INCAR", "KPOINTS", "POSCAR", "POTCAR")
            for f in FILES:
                c.shutil.copy2(
                    local / f,
                    remote / f,
                    direction="put",
                    follow_symlinks=False,
                )

        print(f"running {what}...")
        print(f"{Fore.GREEN}**********************************************************")
        c.subprocess.run(
            [
                "source",
                "/opt/intel/bin/compilervars.sh",
                "-arch",
                "intel64",
                "&&",
                "/home/s/bin/mpirun",
                "-np",
                CORES,
                code,
            ],
            suppress_out=True,
            quiet=True,
            stdout=ProxyStream(
                sys.stdout, local / "relax.out" if what == "QE" else local / "output.txt"
            ),
            stderr=ProxyStream(sys.stderr, local / "error.txt"),
            cwd=remote,
            encoding="utf-8",
        )
        print(f"{Fore.BLUE}**********************************************************")
        print("done")

        print("downloading results")
        c.shutil.download_tree(
            remote, local, exclude="*WAVECAR", quiet=True
        )
