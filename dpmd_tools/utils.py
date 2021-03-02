import atexit
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from shutil import SameFileError, which
from socket import gethostname
from typing import Any, Deque, List
from subprocess import CalledProcessError, run
from getpass import getuser

from ssh_utilities import Connection


def get_graphs(input_graphs: List[str]) -> List[Path]:

    graphs = []
    for graph_str in input_graphs:
        host_path = graph_str.split("@")

        if len(host_path) == 1:
            host = "local"
            path_str = host_path[0]
            local = True
        else:
            host, path_str = host_path
            local = False

        with Connection(host, quiet=True, local=local) as c:

            remote_path = c.pathlib.Path(path_str)
            remote_root = c.pathlib.Path(remote_path.root)
            remote_pattern = str(remote_path.relative_to(remote_path.root))
            remote_graphs = list(remote_root.glob(remote_pattern))

            for rg in remote_graphs:

                print(f"Getting graph from {host}: {rg}")

                local_graph = Path.cwd() / rg.name
                try:
                    c.shutil.copy2(rg, local_graph, direction="get")
                except SameFileError:
                    pass
                graphs.append(local_graph)

    return graphs


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def split_into(n: int, p: int):
    """Split number n to p parts as evenly as possible.

    Example: split_into(32, 3) => [11, 11, 10]
    """
    split = [n / p + 1] * (n % p) + [n / p] * (p - n % p)
    return [int(s) for s in split]


class Loglprint:
    def __init__(self, log_file: Path) -> None:
        log_file.parent.mkdir(exist_ok=True, parents=True)

        self.buffer: Deque[str] = deque()

        if log_file.exists():
            self.log_stream = log_file.open("a")
            fmt = "%d/%m/%Y %H:%M:%S"
            self.__call__("\n=========================================================")
            self.__call__(f"Appending logs: {datetime.now().strftime(fmt)}")
            self.__call__("=========================================================\n")
        else:
            self.log_stream = log_file.open("w")

    def __call__(self, msg: Any) -> Any:
        self.buffer.append(str(msg))
        print(msg)

    def __del__(self):
        self.close()

    def close(self):
        self.log_stream.close()

    def write(self):
        self.log_stream.write("\n".join(self.buffer))
        self.close()


class BlockPBS:

    def __init__(self) -> None:

        script = self._make()
        self.jid = self.qsub(script)
        self._deleted = False
        atexit.register(self.qdel)
        print("successfully set PBS block for 24 hours")

    def __del__(self):
        self.qdel()

    def _make(self) -> Path:

        server = gethostname()

        if server in ("kohn", "planck"):
            ppn = 16
        else:
            ppn = 12

        s = "!/bin/bash\n"
        s += f"#PBS -l nodes={server}:ppn={ppn},walltime=24:00:00\n"
        s += f"#PBS -q batch\n"
        s += f"#PBS -u {getuser()}\n"

        script = Path("pbs.tmp")
        script.write_text(s)

        return script

    @staticmethod
    def qsub(script: Path) -> str:
        qsub = which("qsub")
        if not qsub:
            raise RuntimeWarning("could not get qsub executable")
        else:
            try:
                result = run(
                    [qsub, str(script)], capture_output=True, text=True, cwd=Path.cwd()
                )
                _id = result.stdout.split(".")[1]
            except (IndexError, CalledProcessError):
                raise RuntimeError("could not get PBS job id")
            else:
                return _id

    def qdel(self):
        if self._deleted:
            return

        qdel = which("qdel")
        if not qdel:
            raise RuntimeWarning("could not get qdel executable")
        else:
            result = run(
                [qdel, self.jid], capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.stdout.strip() != "" or result.returncode != 0:
                raise RuntimeError(f"could not delete PBS job {self.jid}")
            else:
                self._deleted = True
                print("PBS block job deleted")