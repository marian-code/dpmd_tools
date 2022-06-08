import re
import sys
from io import TextIOBase
from pathlib import Path
from typing import Optional

from colorama import Fore
from ssh_utilities import Connection, SSHConnection
from tqdm import tqdm

ELECTONIC_STEP = re.compile(r"(?:^RMM|^DAV| {6,8}N)")
WARNING = re.compile(r"([A-Z]+\:)")
ION_STEP = re.compile(r"\s*(\d*)\s*F=\s*")


class ProxyStream:

    def __init__(
        self,
        stream: TextIOBase,
        filename: str,
        filter: bool = False,
        nsw: Optional[int] = None,
    ) -> None:
        self.file = open(filename, "w")
        self.stream = stream
        self.filter = filter
        if filter:
            self.pbar = tqdm(range(nsw), "NSW step", total=nsw, ncols=150)

    def write(self, text: str):

        # must be at the start of the function
        # because we do some colloring for terminal output
        self.file.write(text)

        # print(repr(text), Fore.RED + str(ION_STEP.search(text)) + Fore.RESET)
        # print(repr(text), Fore.RED + str(WARNING.match(text)) + Fore.RESET)
        if self.filter:
            if ELECTONIC_STEP.search(text):
                return
            if WARNING.search(text):
                text = WARNING.sub(Fore.LIGHTYELLOW_EX + r"\g<1>" + Fore.RESET, text)
            if ION_STEP.search(text):
                text = ION_STEP.sub(
                    "NSW step: " + Fore.LIGHTBLUE_EX + r"\g<1>" + Fore.RESET + " F= ",
                    text,
                )
                self.pbar.update(1)
            self.pbar.write(text.rstrip())
        else:
            self.stream.write(text)

    def close(self):
        self.file.close()
        self.pbar.close()


def run_singlepoint(
    *,
    server: str,
    local: Path,
    remote: Path,
    what: str,
    command: Optional[str] = None,
    filter_stream: bool = False,
    nsw: Optional[int] = None,
):
    if filter_stream and what == "QE":
        raise RuntimeError("Stream filtering is not supported for QE")
    elif filter_stream and nsw is None:
        incar = (local / "INCAR").read_text()
        nsw = int(re.findall(r"\s*NSW\s*=\s*(\d*)", incar)[0])


    if server == "fock":
        VASP = "/home/s/Software/vasp.5.4.4_mpi_TS/bin/vasp_std"
    elif server in ("wigner", "bloch", "planck"):
        VASP = "/home/s/Software/VASP/vasp.6.2.1/bin/vasp_std"
    else:
        VASP = "/home/s/Software/VASP/intel-mpi-TS-HI/vasp-5.4.4-TS-HI/bin/vasp_std"

    if server in ("kohn", "planck"):
        CORES = "16"
    elif server in ("wigner", "bloch"):
        CORES = "24"
    else:
        CORES = "12"

    QE = "/home/toth/Software/qe.6.3.0/bin/pw.x < relax.in"

    with Connection(server, local=False, quiet=True) as c:
        c: SSHConnection

        c.os.makedirs(remote, exist_ok=True)

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

        stdout = ProxyStream(
            sys.stdout,
            local / "relax.out" if what == "QE" else local / "output.txt",
            filter=filter_stream,
            nsw=nsw,
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
                "ulimit",
                "-s",
                "unlimited",
                "&&",
                "/home/s/bin/mpirun",
                "-np",
                CORES,
                code,
            ],
            suppress_out=True,
            quiet=True,
            stdout=stdout,
            stderr=ProxyStream(sys.stderr, local / "error.txt"),
            cwd=remote,
            encoding="utf-8",
        )
        stdout.close()
        print(f"{Fore.BLUE}**********************************************************")
        print("done")

        print("downloading results")
        c.shutil.download_tree(remote, local, exclude="*WAVECAR", quiet=True)
