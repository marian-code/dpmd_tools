"""Easily batch recompute large amount of structures on remote clusters.

Supports aurel and all our PCs.
Easily restart and pick up in case of fail.
All is done automatically through ssh, just need to supply structures as list
of ase.Atoms objects.
"""

import itertools
import logging
import re
from abc import abstractmethod
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from datetime import datetime, timedelta
from os import fspath
from pathlib import Path
from shutil import move, rmtree
from subprocess import CompletedProcess
from threading import Lock, Thread
from time import sleep
from typing import (TYPE_CHECKING, ContextManager, Deque, Dict, List, Optional,
                    Tuple, Union)

from ase.atoms import Atoms
from dpmd_tools.recompute.json_serializer import Job, deserialize, serialize
from ssh_utilities import Connection, SSHConnection
from tqdm import tqdm

if TYPE_CHECKING:
    from ssh_utilities import SSHPath

    try:
        from typing import TypedDict  # python > 3.8
    except ImportError:
        from typing_extensions import TypedDict  # python 3.6-7

    CDATA = TypedDict(
        "CDATA",
        {
            "conn": SSHConnection,
            "user": str,
            "name": str,
            "status": List[str],
            "submit": str,
            "max": int,
            "jobs": List[Job],
            "remote_dir": SSHPath,
        },
    )

JOB_ID = re.compile(r"(?:\"|\')(\S+)(?:\"|\')")

log = logging.getLogger(__name__)


# TODO implement local
class RemoteBatchRun:
    """Class that does remote job management heavy lifting so you don't have to.

    Parameters
    ----------
    hosts : List[str]
        list of hosts to recompute on
    start : int
        start index of the list of atoms
    stop : int
        stop index of the list of atoms
    recompute_failed : bool
        whether to retry recomputing jobs failed in previous run
    remote_settings : dict
        settings for remote host dict with max jobs and path to store
        computations
    work_dir : Path
        local working directory
    dump_file : Path
        locatin and name of the restart dump file
    threaded: bool
        if true in critical parts of the code each server interaction is run in separate
        thread, this enhances speed considerably at a small price of unordered output
    """

    # types
    continue_data: Optional[dict] = None
    job_data: Deque[Job]
    c: Dict[str, "CDATA"]
    _LOCK = Union[Lock, ContextManager[None]]

    # classwide variables and constants initailization
    HOUSEKEEPING_INTERVAL: int = 300  # 5 minutes
    _housekeeping_counter: int = 0

    def __init__(
        self,
        hosts: List[str],
        users: List[str],
        start: int,
        stop: int,
        recompute_failed: bool,
        remote_settings: dict,
        work_dir: Path,
        dump_file: Path,
        threaded: bool = False,
    ) -> None:

        # set constants
        self.WD = work_dir
        self.DUMP_FILE = dump_file
        self.SLICE = slice(start, stop)
        self.RECOMPUTE_FAILED = recompute_failed
        self.remote_settings = remote_settings
        self.THREADED = threaded
        if threaded:
            self._LOCK = Lock()
        else:
            self._LOCK = nullcontext()

        (self.WD / "failed").mkdir(exist_ok=True, parents=True)

        self.c = dict()

        log.info("Connecting to requested servers:\n")
        for h, u in zip(hosts, users):
            log.info(f"--> {h}")
            if h == "aurel":
                JOB_STATUS = [
                    "/usr/bin/llq",
                    "-f",
                    r"%id",
                    r"%o",
                    r"%jn",
                    r"%dq",
                    r"%st",
                    r"%p",
                    r"%c",
                    f"-u {u}",
                ]
                JOB_SUBMIT = "/usr/bin/llsubmit"
            elif h == "local":
                JOB_STATUS = [""]
                JOB_SUBMIT = "bash"
            else:
                JOB_STATUS = ["/opt/pbs/bin/qstat"]
                JOB_SUBMIT = "/opt/pbs/bin/qsub"

            self.c[h] = {
                "conn": Connection.get(
                    h,
                    quiet=True,
                    local=True if h == "local" else False,
                    thread_safe=True,
                ),
                "name": h,
                "user": u,
                "status": JOB_STATUS,
                "submit": JOB_SUBMIT,
                "max": remote_settings[h]["max_jobs"],
                "jobs": [],
            }
            # make SSHPath instance
            rm_dir = remote_settings[h]["remote_dir"]
            self.c[h]["remote_dir"] = self.c[h]["conn"].pathlib.Path(rm_dir)

        if self.continue_data:
            self.c.update(self.continue_data)

        # persist data
        self._dump2disk()

        self.total_calc_time: float = 0.0
        self.calc_times_norm: Deque[float] = deque()

        self.computed, self.failed = self.get_finished_jobs(work_dir)

    # * Mandatory override in subclass *************************************************
    @abstractmethod
    def postprocess_job(self, job: Job) -> Optional[float]:
        """Parse job output files and return runtime if job was succesfull or None if not.

        If the method finds that job has failed it should return None and the job will
        be marked as failed.
        """
        raise NotImplementedError("Reimplement in subclass")

    @abstractmethod
    def prepare_calc(
        self, index: int, calc_dir: Path, server: str, atoms: Atoms
    ) -> Tuple[str, str]:
        """Prepade calculation files in specified directory.

        When finished, directory must contain all necessary files to run job
        The method must return job name as it was set in batch script and batch script
        as string.
        """
        raise NotImplementedError("reimplement in subclass")

    # * Optional override in subclass **************************************************
    def set_constants(self, *args, **kwargs):
        """Run this after init so it can set any constants wihout overriding init."""
        raise NotImplementedError("reimplement in subclass")

    def set_job_attr(self, job: Job):
        """Set additional job attributes before it is submitted."""
        pass

    # * persist ************************************************************************
    @classmethod
    def from_json(
        cls,
        hosts: list,
        users: List[str],
        start: int,
        stop: int,
        recompute_failed: bool,
        remote_settings: dict,
        dump_file: Path,
        threaded: bool = False,
    ) -> "RemoteBatchRun":
        """All parameters have the same meaning as in __init__ method.

        All jobs that were running are left as is and finished with previously
        specified parameters!

        If the list of available hosts is shrinked compared to before restart,
        the running jobs are allowed to finish and no new jobs
        are submited to that host.
        """
        if hosts is None:
            hosts = []

        data = deserialize(dump_file, hosts)

        rs = data.pop("remote_settings")
        rs.update(remote_settings)
        work_dir = dump_file.parent

        cls.continue_data = data
        instance = cls(
            hosts,
            users,
            start,
            stop,
            recompute_failed,
            rs,
            work_dir,
            dump_file,
            threaded=threaded,
        )
        instance.computed, instance.failed = cls.get_finished_jobs(work_dir)
        return instance

    def _dump2disk(self):
        self._update_runtimes()
        self.c["remote_settings"] = self.remote_settings
        serialize(self.DUMP_FILE, self.c)
        self.c.pop("remote_settings")

    # * prepare data (optional override in subclass) ***********************************
    @staticmethod
    def get_finished_jobs(work_dir: Path) -> Tuple[List[int], List[int]]:
        """Override to define custom behaviour."""
        log.info("Checking done jobs")
        # get only dirs with calculations
        done_dirs = [
            int(d.name)
            for d in tqdm(work_dir.glob("*/"))
            if d.name.isdigit() and (d / "done").is_file()
        ]

        log.info("Checking failed jobs")
        # get failed dirs
        failed_dirs = [
            int(d.name)
            for d in tqdm((work_dir / "failed").glob("*/"))
            if d.name.isdigit()
        ]

        return done_dirs, failed_dirs

    def get_job_data(self, atoms: List[Atoms]):
        """Override to define custom behaviour."""
        atoms = atoms[self.SLICE]

        # get actually running jobs
        running = []
        for conn in self.c.values():
            running.extend([j.index for j in conn["jobs"]])

        log.info("Collecting jobs data")
        iter_atoms = tqdm(enumerate(atoms, self.SLICE.start + 1))

        data = []
        failed = 0
        for i, a in iter_atoms:

            # check if job was not already run
            if i in self.computed:
                # iter_atoms.write(f"{i} already computed, skipping...")
                continue
            elif i in self.failed:
                failed += 1
                if not self.RECOMPUTE_FAILED:
                    # iter_atoms.write(f"{i} already computed and failed, "
                    #                  f"skipping...")
                    continue
                else:
                    rmtree(self.WD / "failed" / str(i))

            if i in running:
                iter_atoms.write(f"{i} is currently running, skipping...")
                continue

            data.append(Job(index=i, atoms=a, atoms_size=len(a)))

        log.info(f"Found {len(data) + len(running)} jobs to compute.")
        log.info(f"{len(running)} jobs are already scheduled to run or running.")
        if self.RECOMPUTE_FAILED:
            log.info(
                f"Out of that, {failed} jobs are failed and are set to "
                f"be recomputed\n"
            )
        else:
            log.info(f"Also found {failed} failed jobs which will not be recomputed\n")

        # compute total atoms count and save to estimate remainig time
        self.total_atoms = sum([d.atoms_size for d in data])

        self.job_data = deque(sorted(data, key=lambda x: x.atoms_size))

    # * submit jobs ********************************************************************
    def _submit_job(self, server: str, submit_dir: Path) -> Optional[str]:
        jid: Optional[str]

        out = self.c[server]["conn"].subprocess.run(
            [self.c[server]["submit"], "pbs.job"],
            suppress_out=True,
            quiet=True,
            cwd=submit_dir,
            capture_output=True,
            encoding="utf-8",
        )

        # record job id if it needs to be killed
        if server == "aurel":
            try:
                jid = f"{JOB_ID.findall(out.stdout)[0]}.0"
            except IndexError:
                log.warning("Couldn't get job id")
                jid = None
        elif server == "local":
            jid = None
        else:
            jid = out.stdout.split(".")[0]

        return jid

    def _submit2server(self, server: str, free_slots: int, total_jobs: int):

        for _ in range(free_slots):
            # extract connection so we do not need to write long commands
            c = self.c[server]

            # select appropriate job.
            # jobs are sorted in asccending order based on number of atoms.
            # for aurel select from end and for others from begining
            with self._LOCK:
                # pops should be thread safe but dont trust them they probably
                # sometimes mess up
                if server == "aurel":
                    job = self.job_data.pop()
                else:
                    job = self.job_data.popleft()

            log.info(
                f"---------- Optimizing {job.index}: "
                f"{total_jobs - len(self.job_data)}/{total_jobs} "
                f"-------------"
            )

            # prepare calculation in local dir
            # set calc dir and create it
            prepare_dir = self.WD / str(job.index)
            prepare_dir.mkdir(exist_ok=True)
            job_name, job_script = self.prepare_calc(
                job.index, prepare_dir, server, job.pop("atoms")
            )
            (prepare_dir / "pbs.job").write_text(job_script)

            #  for local we don't need to copy
            if server == "local":
                submit_dir = prepare_dir
            else:
                
                submit_dir = c["remote_dir"] / f"{prepare_dir.name}"
                c["conn"].shutil.upload_tree(prepare_dir, submit_dir, quiet=True)

            log.info(f"Scheduling job on {server}")

            # TODO we assume it never fails??
            # run qsub command
            job.id = self._submit_job(server, submit_dir)

            # record remote dir
            job.running_dir = submit_dir

            # record job name
            job.name = job_name

            #  record start time
            job.submit_time = datetime.now()

            # record run time
            job.run_time = 0.0

            # set retry attempts
            job.retry = False

            # record subclass specific attributed for job
            self.set_job_attr(job)

            # put job data in connections dict
            with self._LOCK:
                c["jobs"].append(job)

    def _submit_many(self, free_slots: Counter, total_jobs: int):

        if self.THREADED:
            submit_threads = []
            for i, (server, slots) in enumerate(free_slots.items()):
                t = Thread(
                    target=self._submit2server,
                    args=(server, slots, total_jobs),
                    name=f"submit_{server}_{i}",
                    daemon=True,
                )
                t.start()
                submit_threads.append(t)

            for t in submit_threads:
                t.join()
        else:
            for server, slots in free_slots.items():
                self._submit2server(server, slots, total_jobs)

    def _get_free_slots(self) -> Optional[Counter]:

        server_slots = []
        for server, server_data in self.c.items():
            server_slots.extend(
                [server] * (server_data["max"] - len(server_data["jobs"]))
            )

        if server_slots:
            return Counter(server_slots)
        else:
            return None

    # *get jobs back *******************************************************************
    def _download_job(self, server: str, conn: SSHConnection, job: Job):

        #  no need to download if we are on
        if server == "local":
            return True

        # TODO ideally skip WAVECAR too
        # on occasion fails, do 3 tries
        for _ in range(3):
            try:
                conn.shutil.download_tree(
                    job.running_dir,
                    self.WD / job.running_dir.name,
                    exclude="*POTCAR",
                    quiet=True,
                    remove_after=True,  # TODO problem with permission with PBS created dirs, they cannot be removed
                )
            except OSError as e:
                log.debug(f"error when downloading job {job.index}: {e}")
            else:
                return True

        log.warning(f"could not download job {job.index}, marking as failed")
        (self.WD / "failed" / job.running_dir.name).mkdir(exist_ok=True)

        return False

    def _get_server_jobs(self, server: str, done: Deque[int]):

        data = self.c[server]

        # pop largest indices first otherwise they will get messed up
        for d in sorted(done, reverse=True):

            job = data["jobs"].pop(d)

            log.info(f"Retrieving finished job {job.index} from {server}")

            if not self._download_job(server, data["conn"], job):
                continue
            else:
                calc_time = self.postprocess_job(job)

                if not calc_time:
                    calc_dir = self.WD / job.running_dir.name
                    log.warning(f"Computation {job.index} failed !!!")
                    rmtree(self.WD / "failed" / calc_dir.name, ignore_errors=True)
                    move(fspath(calc_dir), fspath(self.WD / "failed" / calc_dir.name))
                    calc_time = 0.0
                else:
                    log.info(
                        f"Current computation CPU time: {timedelta(seconds=calc_time)}"
                    )

                with self._LOCK:
                    self.total_calc_time += calc_time
                    self.calc_times_norm.append(calc_time / job.atoms_size)
                    self.total_time += job.run_time / job.atoms_size

    def _get_finished_jobs(self, done_jobs: Dict[str, Deque[int]]):

        if self.THREADED:
            threads = []
            for server, done in done_jobs.items():
                t = Thread(
                    target=self._get_server_jobs,
                    args=(server, done),
                    daemon=True,
                    name=f"get_{server}_done",
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        else:
            for server, done in done_jobs.items():
                self._get_server_jobs(server, done)

        self._dump2disk()
        self._time_statistics()

    def _get_done_indices(self) -> Dict[str, Deque[int]]:

        done_jobs: Dict[str, Deque[int]] = defaultdict(deque)
        for server, data in self.c.items():
            for i, job in enumerate(data["jobs"]):
                if data["conn"].os.isfile(job.running_dir / "done"):
                    done_jobs[server].append(i)

        return done_jobs

    # * housekeeping *******************************************************************
    def _job_status(self, server: str) -> CompletedProcess:

        #  sometimes fails and output is empty for no apparent reason, loop until OK
        while True:
            out = self.c[server]["conn"].subprocess.run(
                self.c[server]["status"],
                suppress_out=False,
                capture_output=True,
                encoding="utf-8",
            )
            if out.stdout.strip() and not out.stderr and out.returncode == 0:
                break

        return out

    def _housekeeping(self):
        """Take care of stuck/failed to submit jobs."""
        log.info("Running sheduled job housekeeping to clean up stuck jobs")
        for server, data in self.c.items():

            if server == "local":
                log.debug("skipping housekeeping for local jobs")
                continue

            out = self._job_status(server)

            delete_jobs = []
            for i, job in enumerate(data["jobs"]):
                # test if job id is present in task scheduler output
                if job.id and job.id in out.stdout:
                    continue
                # sometimes job id is not captured when submiting, in that case used
                # job name instead
                elif job.name in out.stdout:
                    continue
                # one last check if done file is present
                elif data["conn"].os.isfile(job.running_dir / "done"):
                    continue
                # now check if job was already resubmited, if yes consider it failed
                elif job.retry:
                    # if job was already resubmited discard it as failed
                    # TODO potentialy check for error file but needs to know its name?!
                    log.warning(
                        f"Job {job.index} on {server} did not complete even after "
                        f"resubmission, downloading and marking as failed"
                    )
                    self._download_job(server, data["conn"], job)
                    delete_jobs.append(i)
                # if all contitions pass this means that the job is somehow stuck and
                # is blocking the queue try to resubmit it
                else:
                    log.info(
                        f"Resubmmiting job {job.index} on "
                        f"{server} it appears to be stuck"
                    )
                    # if all failed resubmit
                    job.id = self._submit_job(server, job.running_dir)
                    job.retry = True
                    job.submit_time = datetime.now()

            for i in sorted(delete_jobs, reverse=True):
                del data["jobs"][i]

    def _update_runtimes(self):

        now = datetime.now()

        for data in self.c.values():
            for job in data["jobs"]:
                job.run_time = (now - job.submit_time).total_seconds()

    # * run ****************************************************************************
    def _wait(self):

        wait_time = 0
        wait_loop = True

        while wait_loop:

            done_jobs = self._get_done_indices()
            if done_jobs:
                self._get_finished_jobs(done_jobs)
                wait_loop = False
            else:
                print(
                    f"Waiting for jobs completition: "
                    f"{str(timedelta(seconds=wait_time))}",
                    end="\r",
                )
                # TODO this runs too often
                self._housekeeping_counter += 1

            self._update_runtimes()
            if self._housekeeping_counter != 0 and self._housekeeping_counter % self.HOUSEKEEPING_INTERVAL == 0:
                self._housekeeping()

            # sleep 5 seconds before next check
            sleep(1)
            wait_time += 1

    def loop(self):
        """Run computational loop."""
        self.total_time = 0
        total_jobs = len(self.job_data)

        while self.job_data:

            self._job_satistics()

            # select server
            free_slots = self._get_free_slots()
            if not free_slots:
                self._wait()
                continue
            else:
                self._submit_many(free_slots, total_jobs)

            # save data to disk
            self._dump2disk()

        # wait until all jobs are finished
        log.info("All jobs have been submitted, waiting for completition")
        while True:

            remaining = sum([len(v["jobs"]) for v in self.c.values()])
            log.info(f"waiting for last {remaining} jobs to finish")

            # count running jobs
            if remaining == 0:
                break
            else:
                self._wait()

        log.info("Job done, cleaning up.")
        self.DUMP_FILE.unlink()

    # * utilities **********************************************************************
    def _job_satistics(self):

        # output running jobs statistics
        log.info("Job statistics:")

        to_delete = []
        for k, v in self.c.items():
            if len(v["jobs"]) == 0 and v["max"] == 0:
                to_delete.append(k)
            elif v["max"] == 0:
                log.info(
                    f"{k:<11}: {len(v['jobs']):>2} job finishing, "
                    f"no further jobs will be submited"
                )
                log.info(f"-> ids     : {v['jobs'][0].index}")
            else:
                jobs = sorted(set([j.index for j in v["jobs"]]))
                # show contracted list of intervals, not each job
                job_intervals = []
                for _, group in itertools.groupby(
                    enumerate(jobs), lambda t: t[1] - t[0]
                ):
                    group = list(group)
                    if group[0][1] == group[-1][1]:
                        job_intervals.append(str(group[0][1]))
                    else:
                        job_intervals.append(f"{group[0][1]}-{group[-1][1]}")

                log.info(f"{k:<11}: {len(v['jobs']):>2}/{v['max']:<2} running")
                log.info(f"-> ids     : {', '.join(job_intervals)}")

        # delete host which will not receive further jobs
        for td in to_delete:
            self.c.pop(td)

        self._dump2disk()

    def _time_statistics(self):
        if len(self.calc_times_norm) == 0:
            cpu_avg = 0.0
        else:
            cpu_avg = sum(self.calc_times_norm) / len(self.calc_times_norm)

        left_atoms = sum([j.atoms_size for j in self.job_data])
        cpu_eta = cpu_avg * left_atoms

        log.info(f"Average CPU time/atom: {timedelta(seconds=cpu_avg)}")
        log.info(f"Estimated CPU time left: {timedelta(seconds=cpu_eta)}")
        log.info(
            f"Total CPU time from start: {timedelta(seconds=self.total_calc_time)}"
        )

        if self.total_atoms - left_atoms == 0:
            time_avg = 0.0
        else:
            time_avg = self.total_time / (self.total_atoms - left_atoms)
        time_eta = left_atoms * time_avg

        log.info(f"Average time/atom: {timedelta(seconds=time_avg)}")
        log.info(f"Estimated time left: {timedelta(seconds=time_eta)}")
        log.info(f"Total time from start: {timedelta(seconds=self.total_time)}")

        log.info("--------------------------------------------------------------------")

    def __del__(self):
        for host in self.c.values():
            host["conn"].close(quiet=True)

    def handle_ctrl_c(self, sig, frame):
        self._dump2disk()
        log.info("Aborted by user, exiting... Please allow a few seconds to finish")
        assert False, "abborted by user"
