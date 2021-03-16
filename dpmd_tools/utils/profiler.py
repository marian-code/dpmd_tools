"""Profiling module."""
from contextlib import contextmanager
from pathlib import Path
import logging

__all__ = ["ContextProfiler", "init_yappi"]

log = logging.getLogger(__name__)


def finish_yappi():
    """Stop profiler, collect stats and write them to disc."""
    import yappi

    # define directory for stats save and ensure it exists
    OUT_FILE = Path.cwd() / "profile_data"
    OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

    # stop yappi profiler
    log.info('[YAPPI STOP]')
    yappi.stop()

    # get yappi function stats
    log.info('[YAPPI WRITE]')
    stats = yappi.get_func_stats()

    # write different formats of functions statistics
    for stat_type in ['pstat', 'callgrind', 'ystat']:
        path = OUT_FILE.with_suffix(f".{stat_type}")
        log.info(f'writing {path}')
        stats.save(path, type=stat_type)

    # write summary function statistics
    path = OUT_FILE.with_suffix(".func_stats")

    log.info('\n[YAPPI FUNC_STATS]')
    log.info(f"writing {path}")

    with path.open("w") as fh:
        stats.print_all(out=fh)

    # write thread based statistics
    path = OUT_FILE.with_suffix(".thread_stats")

    log.info('\n[YAPPI THREAD_STATS]')
    log.info(f"writing {path}")

    with path.open("w") as fh:
        yappi.get_thread_stats().print_all(out=fh)

    log.info('[YAPPI OUT]')


@contextmanager
def ContextProfiler():  # NOSONAR
    """Context profiler using yappi."""
    import yappi

    log.info('[YAPPI START]')
    yappi.set_clock_type('wall')
    yappi.start()

    try:
        yield None
    finally:
        finish_yappi()


def init_yappi(write_at_exit: bool = True):
    """Initialize yappi profiler and register atexit handler.

    Stats are written automatically on application exit.
    """
    import yappi
    import atexit

    log.info('[YAPPI START]')
    yappi.set_clock_type("cpu")  # 'wall')
    yappi.start()

    if write_at_exit:
        atexit.register(finish_yappi)
