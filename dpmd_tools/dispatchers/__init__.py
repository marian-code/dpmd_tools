from typing import Dict, TYPE_CHECKING, Callable
import logging
from .ibm import batch_script_ibm
from .pbs import batch_script_pbs

if TYPE_CHECKING:

    DISP = Callable[[str, int, str, bool, bool, int], str]
    DECORATED = Callable[[int, str, bool, bool, int], str]
    DECORATOR = Callable[[DISP], DISP]

log = logging.getLogger(__name__)


class Dispatcher:
    _funcs: Dict[str, "DECORATED"] = {}

    @classmethod
    def register(cls, fmt: str) -> "DECORATOR":
        """Can be used as decorator or in function style as shown below.""" 
        def decorator(func) -> "DISP":
            log.debug(f"registered dispatcher function for {fmt}")
            cls._funcs[fmt] = Decorated(fmt, func)
            return func

        return decorator

    @classmethod
    def get(cls, fmt: str) -> "DECORATED":
        return cls._funcs[fmt]


class Decorated:
    def __init__(self, server: str, decorated: "DISP") -> None:
        self._server = server
        self._decorated = decorated

    def __call__(
        self,
        n_nodes: int,
        ident: str,
        scan: bool,
        priority: bool = True,
        hour_length: int = 12,
    ) -> str:
        job = self._decorated(
            self._server, n_nodes, ident, scan, priority, hour_length
        )

        job += "\ntouch done"
        return job


Dispatcher.register("aurel")(batch_script_ibm)
Dispatcher.register("kohn")(batch_script_pbs)
Dispatcher.register("fock")(batch_script_pbs)
Dispatcher.register("hartree")(batch_script_pbs)
Dispatcher.register("landau")(batch_script_pbs)
Dispatcher.register("schrodinger")(batch_script_pbs)
Dispatcher.register("planck")(batch_script_pbs)
