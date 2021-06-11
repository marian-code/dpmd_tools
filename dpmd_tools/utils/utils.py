import atexit
import math
from collections import deque
from datetime import datetime
from getpass import getuser
from pathlib import Path
from shutil import SameFileError, which
from socket import gethostname
from subprocess import CalledProcessError, run
from typing import Any, Deque, List
from warnings import warn

from colorama import Fore
from ssh_utilities import Connection


def get_remote_files(
    input_paths: List[str], remove_after: bool = False, same_names: bool = False
) -> List[Path]:

    graphs = []
    for path_str in input_paths:
        host_path = path_str.split("@")

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
            remote_files = list(remote_root.glob(remote_pattern))

            for rf in remote_files:

                print(f"Getting file from {host}: {rf}")

                if same_names and not local:
                    local_graph = Path.cwd() / f"{rf.stem}.{rf.parent.name}{rf.suffix}"
                else:
                    local_graph = Path.cwd() / rf.name
                try:
                    c.shutil.copy2(rf, local_graph, direction="get")
                except SameFileError:
                    pass
                graphs.append(local_graph)

    if remove_after:

        def _remove_graphs(graphs: List[Path]):
            for g in graphs:
                g.unlink()

        atexit.register(_remove_graphs, graphs)

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
        print(msg)

        m = str(msg)
        for color in vars(Fore).values():
            m = m.replace(color, "")
        self.buffer.append(m)

    def __del__(self):
        self.close()

    def close(self):
        self.log_stream.close()

    def write(self):
        self.log_stream.write("\n".join(self.buffer))
        self.close()


class BlockPBS:

    _deleted = False

    def __init__(self) -> None:

        self.script = self._make()
        self.jid = self.qsub(self.script)
        atexit.register(self.qdel)
        print("successfully set PBS block for 24 hours")

    def __del__(self):
        self.qdel()

    def _make(self) -> Path:

        server = gethostname().lower()

        if server in ("kohn", "planck"):
            ppn = 6
        else:
            ppn = 6

        s = "#!/bin/bash\n"
        s += f"#PBS -l nodes={server.lower()}:ppn={ppn},walltime=24:00:00\n"
        s += f"#PBS -q batch\n"
        s += f"#PBS -k n\n"  # force PBS to delete stdout and stderr files
        s += f"#PBS -u {getuser()}\n"
        s += f"#PBS -N GPU-block-job\n"
        s += f"sleep 24h"

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
                _id = result.stdout.split(".")[0]
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
                warn(f"could not delete PBS job {self.jid}", UserWarning)
            else:
                self.script.unlink()
                self._deleted = True
                print("PBS block job deleted")
