import json
import logging
from datetime import datetime
from functools import singledispatch
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, Optional

from ssh_utilities import Connection, SSHConnection

if TYPE_CHECKING:
    from ssh_utilities import SSHPath

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

log = logging.getLogger(__name__)


class Job:
    index: int
    atoms_size: int
    id: Optional[str]
    running_dir: "SSHPath"
    name: str
    SCAN: bool
    submit_time: datetime
    run_time: float = 0.0
    retry: bool = False

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def pop(self, key: str):
        try:
            return getattr(self, key)
        finally:
            delattr(self, key)


@singledispatch
def json_serializable(obj):
    raise TypeError(f"{type(obj)} serialization is not supported.")


@json_serializable.register(SSHConnection)
def _handle_conn(obj: SSHConnection):
    return str(obj)


@json_serializable.register(Job)
def _handle_job(obj: Job):
    return vars(obj)


@json_serializable.register(datetime)
def _handle_datetime(obj: datetime):
    return obj.strftime(TIME_FORMAT)


@json_serializable.register(Path)
def _handle_path(obj):
    try:
        return str(obj.resolve())
    except Exception as e:
        log.warning(e)
        return str(obj)


def serialize(path: Path, data: dict):

    with path.open("w") as f:
        json.dump(data, f, indent=4, sort_keys=True, default=json_serializable)


def deserialize(path: Path, hosts: Sequence[str]) -> dict:

    with path.open() as f:
        data = json.load(f)

        for host, conn in data.items():
            if host == "remote_settings":
                continue
            conn["conn"] = Connection.from_str(conn["conn"], quiet=True)
            if host not in hosts:
                conn["max"] = 0

            # cast path to SSHPath
            conn["remote_dir"] = conn["conn"].pathlib.Path(conn["remote_dir"])

            for i, job in enumerate(conn["jobs"]):
                job = Job(**job)
                conn["jobs"][i] = job
                job.running_dir = conn["conn"].pathlib.Path(job.running_dir)
                job.submit_time = datetime.now()

        return data
