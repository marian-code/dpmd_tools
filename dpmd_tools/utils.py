from pathlib import Path
from shutil import SameFileError
from typing import Deque, List, Any
from datetime import datetime
from collections import deque
import math

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
