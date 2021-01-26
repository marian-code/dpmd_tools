"""Easily batch recompute large amount of structures on remote clusters.

Supports aurel and all our PCs.
Easily restart and pick up in case of fail.
All is done automatically through ssh, just need to supply structures as list
of ase.Atoms objects.
"""

import argparse
import itertools
import json
import logging
import re
from collections import deque
from datetime import timedelta
from functools import singledispatch
from os import fspath
from pathlib import Path
from shutil import copy, move, rmtree
from time import sleep, time
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.calculators.vasp import Vasp
from ase.io import write
from ase.io.extxyz import read_xyz
from ssh_utilities import Connection, SSHConnection
from ssh_utilities.exceptions import CalledProcessError
from tqdm import tqdm

if TYPE_CHECKING:
    from ssh_utilities import SSHPath
    try:
        from typing import TypedDict  # python > 3.8
    except ImportError:
        from typing_extensions import TypedDict  # python 3.6-7

    JOB = TypedDict(
        "JOB",
        {
            "index": int,
            "atoms_size": int,
            "job_id": str,
            "running_dir": SSHPath,
            "job_name": str,
            "SCAN": bool
        },
        total=False
    )

    CDATA = TypedDict(
        "CDATA",
        {
            "conn": SSHConnection,
            "name": str,
            "status": List[str],
            "submit": str,
            "max": int,
            "jobs": List[JOB],
            "remote_dir": SSHPath
        },
        total=False)


CPU_TIME = re.compile(r"\s+Total CPU time used \(sec\):\s+(\S+)\s*")
JOB_ID = re.compile(r"(?:\"|\')(\S+)(?:\"|\')")

log = logging.getLogger(__name__)


def input_parser():
    p = argparse.ArgumentParser(
        description="script to recompute arbitrarry atoms set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("-s", "--start", help="start of the interval",
                   type=int, default=0)
    p.add_argument("-e", "--end", help="end of the interval (None = end)",
                   type=int, default=None)
    p.add_argument("-r", "--remote", help="server to run on", nargs="+",
                   required=True, choices=("aurel", "kohn", "schrodinger",
                   "fock", "hartree", "landau"))
    p.add_argument("-S", "--SCAN", help="whether to use SCAN functional",
                   type=bool, default=False)
    p.add_argument("-f", "--failed-recompute", help="re-run failed jobs",
                   action="store_true", default=False)

    return vars(p.parse_args())


class Recompute:

    continue_data: Optional[dict] = None
    job_data: Deque["JOB"]
    c: Dict[str, "CDATA"]

    def __init__(self, hosts: List[str], start: int, stop: int,
                 recompute_failed: bool, scan: bool, remote_settings: dict,
                 work_dir: Path, data_dir: Path, dump_file: Path,
                 incar_template: Optional[Path] = None,
                 target_KP_density: Optional[float] = None,
                 ) -> None:

        # set constants
        self.WD = work_dir
        self.DATA_DIR = data_dir
        self.DUMP_FILE = dump_file
        self.SCAN = scan
        self.SLICE = slice(start, stop)
        self.RECOMPUTE_FAILED = recompute_failed
        self.remote_settings = remote_settings
        self.TARGET_KP_DENSITY = target_KP_density
        if incar_template:
            self.INCAR_TEMPLATE = Vasp(xc="PBE")
            self.INCAR_TEMPLATE.read_incar(incar_template)
        else:
            self.INCAR_TEMPLATE = None

        if target_KP_density:
            log.info(f"Targeting K point density {target_KP_density}")

        (self.WD / "failed").mkdir(exist_ok=True, parents=True)

        self.c = dict()

        log.info("\nConnecting to requested servers:")
        for h in hosts:
            log.info(f"--> {h}")
            if h == "aurel":
                JOB_STATUS = ["/usr/bin/llq", "-f", r"%id", r"%o", r"%jn",
                              r"%dq", r"%st", r"%p", r"%c"]
                JOB_SUBMIT = "/usr/bin/llsubmit"
            else:
                JOB_STATUS = ["/opt/pbs/bin/qstat"]
                JOB_SUBMIT = "/opt/pbs/bin/qsub"

            self.c[h] = {
                "conn": Connection.get(h, quiet=True, local=False,
                                       thread_safe=False),
                "name": h,
                "status": JOB_STATUS,
                "submit": JOB_SUBMIT,
                "max": remote_settings[h]["max_jobs"],
                "jobs": []
            }
            # make SSHPath instance
            rm_dir = remote_settings[h]["remote_dir"]
            self.c[h]["remote_dir"] = self.c[h]["conn"].pathlib.Path(rm_dir)

        if self.continue_data:
            self.c.update(self.continue_data)

        # persist data
        self._dump2disk()

        self.total_calc_time: float = 0.0
        self.calc_times_norm: List[float] = []

        self.computed, self.failed = self.get_finished_jobs()

    def __del__(self):
        for host in self.c.values():
            host["conn"].close(quiet=True)

    @classmethod
    def from_json(cls, hosts: list, start: int, stop: int,
                  recompute_failed: bool, scan: bool, data_dir: Path,
                  dump_file: Path, target_KP_density: Optional[float] = None
                  ) -> "Recompute":
        """All parameters have the same meaning as in __init__ method.

        All jobs that were running are left as is and finished with previously
        specified parameters!

        If the list of available hosts is shrinked compared to before restart,
        the running jobs are allowed to finish and no new jobs
        are submited to that host.
        """
        if hosts is None:
            hosts = []

        with dump_file.open() as f:
            data = json.load(f)

        remote_settings = data.pop("remote_settings")
        work_dir = dump_file.parent

        for host, conn in data.items():
            conn["conn"] = Connection.from_str(conn["conn"], quiet=True)
            if host not in hosts:
                conn["max"] = 0

            # cast path to SSHPath
            conn["remote_dir"] = conn["conn"].pathlib.Path(conn["remote_dir"])

            for job in conn["jobs"]:
                job["running_dir"] = conn["conn"].pathlib.Path(job["running_dir"])

        cls.continue_data = data
        return cls(hosts, start, stop, recompute_failed, scan, remote_settings,
                   work_dir, data_dir, dump_file,
                   target_KP_density=target_KP_density)

    def get_finished_jobs(self) -> Tuple[List[int], List[int]]:
        """Override to define custom behaviour."""
        log.info("Checking done jobs")

        # get only dirs with calculations
        done_dirs = [int(d.name) for d in self.WD.glob("*/")
                     if d.name.isdigit()]

        # get failed dirs
        failed_dirs = [int(d.name) for d in (self.WD / "failed").glob(("*/"))
                       if d.name.isdigit()]

        return done_dirs, failed_dirs

    def _dump2disk(self):

        @singledispatch
        def json_serializable(obj):
            raise TypeError(f"{type(obj)} serialization is not supported.")

        @json_serializable.register(SSHConnection)
        def _handle_conn(obj):
            return str(obj)

        @json_serializable.register(Path)
        def _handle_path(obj):
            try:
                return str(obj.resolve())
            except Exception as e:
                log.warning(e)
                return str(obj)

        self.c["remote_settings"] = self.remote_settings

        with self.DUMP_FILE.open("w") as f:
            json.dump(self.c, f, indent=4, sort_keys=True,
                      default=json_serializable)

        self.c.pop("remote_settings")

    def get_job_data(self, atoms: List[Atoms]):
        """Override to define custom behaviour."""
        atoms = atoms[self.SLICE]

        # get actually running jobs
        running = []
        for conn in self.c.values():
            running.extend([j["index"] for j in conn["jobs"]])

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
                    rmtree(WORK_DIR / "failed" / str(i))

            if i in running:
                iter_atoms.write(f"{i} is currently running, skipping...")
                continue

            data.append({
                "index": i,
                "atoms": a,
                "atoms_size": len(a)
            })

        log.info(f"Found {len(data) + len(running)} jobs to compute.")
        log.info(f"{len(running)} jobs are already scheduled to run or "
                 f"running.")
        if self.RECOMPUTE_FAILED:
            log.info(f"Out of that, {failed} jobs are failed and are set to "
                     f"be recomputed\n")
        else:
            log.info(f"Also found {failed} failed jobs which will not be "
                     f"recomputed\n")

        # compute total atoms count and save to estimate remainig time
        self.total_atoms = sum([d["atoms_size"] for d in data])

        self.job_data = deque(sorted(data, key=lambda x: x["atoms_size"]))

    @staticmethod
    def _get_cpu_time(calc_dir, filename: str) -> float:

        output = (calc_dir / filename).read_text()
        return float(CPU_TIME.findall(output)[0])

    def _get_done_jobs(self, server: str, done: list):

        data = self.c[server]

        # pop largest indices first otherwise they will get messed up
        for d in sorted(done, reverse=True):

            job = data["jobs"].pop(d)
            remote_dir = job["running_dir"]
            calc_dir = self.WD / remote_dir.name

            log.info(f"Retrieving finished job {job['index']} from {server}")

            data["conn"].shutil.download_tree(
                remote_dir, calc_dir, exclude="*WAVECAR",
                quiet=True, remove_after=True
            )

            try:
                calc_time = self._get_cpu_time(calc_dir, "OUTCAR")

                if job["SCAN"]:
                    calc_time += self._get_cpu_time(calc_dir, "OUTCAR0")

            except (IndexError, FileNotFoundError):
                log.warning(f"Computation {job['index']} failed !!!")
                rmtree(self.WD / "failed" / calc_dir.name,
                       ignore_errors=True)
                move(fspath(calc_dir),
                     fspath(self.WD / "failed" / calc_dir.name))
                calc_time = 0.0
            else:
                log.info(f"Current computation CPU time: "
                         f"{timedelta(seconds=calc_time)}")
            finally:
                self.total_calc_time += calc_time
                self.calc_times_norm.append(calc_time / job["atoms_size"])

                self._dump2disk()

        if len(self.calc_times_norm) == 0:
            cpu_avg = 0.0
        else:
            cpu_avg = sum(self.calc_times_norm) / len(self.calc_times_norm)

        left_atoms = sum([j["atoms_size"] for j in self.job_data])
        cpu_eta = cpu_avg * left_atoms

        log.info(f"Average CPU time/atom: "
                 f"{timedelta(seconds=cpu_avg)}")
        log.info(f"Estimated CPU time left: "
                 f"{timedelta(seconds=cpu_eta)}")
        log.info(f"Total CPU time from start: "
                 f"{timedelta(seconds=self.total_calc_time)}")
        
        total_time = time() - self.start_time
        if self.total_atoms - left_atoms == 0:
            time_avg = 0.0
        else:
            time_avg = total_time / (self.total_atoms - left_atoms)
        time_eta = left_atoms * time_avg
        
        log.info(f"Average time/atom: "
                 f"{timedelta(seconds=time_avg)}")
        log.info(f"Estimated time left: "
                 f"{timedelta(seconds=time_eta)}")
        log.info(f"Total time from start: "
                 f"{timedelta(seconds=total_time)}")
        
        log.info("------------------------------"
                 "----------------------------------\n")

    def _wait(self):

        wait_time = 0
        wait_loop = True

        while wait_loop:
            for server, data in self.c.items():

                jobs = data["jobs"]
                done = []
                for i, job in enumerate(jobs):
                    if data["conn"].os.isfile(job["running_dir"] / "done"):
                        done.append(i)

                try:
                    out = data["conn"].subprocess.run(
                        data["status"], suppress_out=False,
                        capture_output=True, check=True, encoding="utf-8"
                    )
                except CalledProcessError:
                    print("CalledProcessError capturing qstat out", end="\r")
                    continue

                if not out:
                    log.warning("Output is None!")
                    continue

                # if done file for any job is present in its directory
                # break loop
                if done:
                    self._get_done_jobs(server, done)
                    wait_loop = False
                    break

                # check if error occured
                elif out.stderr:
                    log.warning(f"error capturing qstat out {out.stderr}")
                    continue

                # check if all job names are still in queue
                elif all(j["job_name"] in out.stdout for j in jobs):
                    print(f"Waiting for jobs completition: "
                          f"{str(timedelta(seconds=wait_time))}", end="\r")

                # some unspecified error occured,
                # probably wrong capture of stdout
                else:
                    print("Unspecified error in reading output",
                          " " * 100, end="\r")
                    # TODO stdout and stderr are empty and returncode is 1 or 0
                    # log.info(out)

            # sleep 5 seconds before next check
            sleep(5)
            wait_time += 5

        print(" " * 100)

        return

    def get_incar(self, server: str, atoms: Atoms, calc_dir: Path):

        args = {
            "nsim": 4,
            "npar": 4,
            "kpar": 2 if len(atoms) > 10 else 1
        }

        self.INCAR_TEMPLATE.set(**args)
        try:
            self.INCAR_TEMPLATE.write_incar(atoms)
        finally:
            move(fspath(WORK_DIR / "INCAR"), calc_dir)

    def get_batch_script(self, server: str, atoms: Atoms, ident: str,
                         priority: bool = True, hour_length: int = 12):

        s = ""
        if server == "aurel":
            if hour_length <= 12:
                cls = "short"
            elif hour_length <= 48:
                cls = "medium"
            else:
                raise ValueError(f"job time {hour_length} is over limit of 48")

            prio = "_priority" if priority else ""

            if len(atoms) < 10:
                nodes = 1
            else:
                nodes = 2

            s += "# File generated by python recompute script"
            s += "# @ account_no = martonak-407"
            s += "# @ output = output.txt"
            s += "# @ error = error.txt"
            s += f"# @ cpu_limit = {hour_length}:00:00"
            s += "# @ job_type = parallel"
            s += f"# @ job_name = {ident}"
            s += f"# @ class = {cls}{prio}"
            s += f"# @ node = {nodes}"
            s += "# @ tasks_per_node = 32"
            s += "# @ network.MPI = sn_all,not_shared,US"
            s += "# @ notification = always"
            s += "# @ environment = COPY_ALL"
            s += "# @ rset = RSET_MCM_AFFINITY"
            s += "# @ mcm_affinity_options = mcm_mem_req mcm_distribute mcm_sni_none"
            s += "# @ task_affinity = core(1)"
            s += "# @ queue"

            if self.SCAN:
                s+= "mpiexec /gpfs/home/dominika/vasp.5.3.2.complex.27.10.14"
            else:
                s += ("mpiexec /gpfs/home/kohulak/Software/vasp.5.4.4-testing/"
                      "bin/vasp_std")

        else:
            if server == "kohn":
                ppn = 16
            else:
                ppn = 12

            s += "!/bin/bash\n"
            s += (f"#PBS -l nodes={server}:ppn={ppn},walltime="
                  f"{hour_length}:00:00\n")
            s += f"#PBS -q batch\n"
            s += f"#PBS -u rynik\n"
            s += f"#PBS -N {ident}\n"
            s += f"#PBS -e error.txt\n"
            s += f"#PBS -o output.txt\n"
            s += f"#PBS -V\n\n"
            s += "NAME=$(echo $(hostname | tr '[:upper:]' '[:lower:]'))\n\n"
            s += 'if [ "$NAME" = "fock" ]\n'
            s += "then\n"
            s += '    VASP="/home/s/Software/vasp.5.4.4_mpi_TS/bin/vasp_std"\n'
            s += 'else\n'
            s += ('    VASP="/home/s/Software/VASP/intel-mpi-TS-HI/'
                  'vasp-5.4.4-TS-HI/bin/vasp_std"\n')
            s += "fi\n\n"
            s += "cd $PBS_O_WORKDIR\n"
            s += "source /opt/intel/bin/compilervars.sh intel64\n"
            s += 'export PATH="$PATH:/opt/intel/bin"\n'
            s += ('export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/mkl/'
                  'lib/intel64_lin"\n')
            s += f"/home/s/bin/mpirun -np {ppn} $VASP"

        s += "touch done"
        return s

    def prepare_calc(self, index: int, server: str, atoms: Atoms
                     ) -> Tuple[Path, str]:

        log.info("Preparing job on local side...")

        job_name = f"Label-{index}"

        # set calc dir and create it
        calc_dir = self.WD / str(index)

        calc_dir.mkdir()

        # copy potential file
        copy(self.DATA_DIR / "POTCAR", calc_dir)

        # copy kpoints file
        if self.TARGET_KP_DENSITY:
            log.info(f"setting KPOINTS so the target density is close to "
                     f"{self.TARGET_KP_DENSITY}")
            (calc_dir / "KPOINTS").write_text(self.generate_kpoints(atoms))
        else:
            log.info("copying KPOINS from inputs directory")
            copy(self.DATA_DIR / "KPOINTS", calc_dir)

        # copy first incar file
        if self.INCAR_TEMPLATE:
            log.info("INCAR and batch script will be generated automatically")
            self.get_incar(server, atoms, calc_dir)
        else:
            log.info("using template files for INCAR and batch script")
            copy(self.DATA_DIR / f"INCAR1_{server}", calc_dir / "INCAR")

        if self.SCAN:
            log.info("Running SCAN calcualtion")
            copy(self.DATA_DIR / "INCAR2", calc_dir)
            pbs = (self.DATA_DIR / f"pbs_SCAN_{server}.job")
        else:
            log.info("Running PBE calcualtion")
            pbs = (self.DATA_DIR / f"pbs_PBE_{server}.job")

        # write identification number and
        # copy PBS submit script to calc dir
        if self.INCAR_TEMPLATE:
            job_script = self.get_batch_script(server, atoms, job_name)
        else:
            job_script = pbs.read_text().replace("GAP_IDENT", job_name)
        
        (calc_dir / "pbs.job").write_text(job_script)

        # write structure to file
        # pop is used because we do not need to keep the structure
        write(str(calc_dir / "POSCAR"), atoms, direct=True, vasp5=True,
              ignore_constraints=True,
              label='File generated by python recompute script')

        return calc_dir, job_name

    def generate_kpoints(self, atoms: Atoms) -> str:

        KP_STR = (
            "KPOINTS adjusted to retain same density as clathrate\n"
            "0\n"
            "M\n"
            "{} {} {}\n"
            "0 0 0\n"
        )
        POINTS = np.power(np.arange(15), 3)

        # get reciprocal volume
        vol = 1 / atoms.get_volume()
        # get num. of kpoints
        kpoints = vol / self.TARGET_KP_DENSITY
        diff_points = (POINTS - kpoints)

        # get closest
        k = np.where(diff_points > 0, diff_points, np.inf).argmin()
        k = k if pow(k, 3) > kpoints else k + 1

        log.info(f"final KPOINT configuration is: {k}x{k}x{k}")
        return KP_STR.format(k, k, k)

    def _job_satistics(self):

        # output running jobs statistics
        log.info("Job statistics:")

        to_delete = []
        for k, v in self.c.items():
            if len(v['jobs']) == 0 and v["max"] == 0:
                to_delete.append(k)
            elif v["max"] == 0:
                log.info(f"{k:<11}: {len(v['jobs']):>2} job finishing, "
                         f"no further jobs will be submited")
                log.info(f"-> ids     : {v['jobs'][0]['index']}")
            else:
                jobs = sorted(set([j['index'] for j in v['jobs']]))
                # show contracted list of intervals
                job_intervals = []
                for _, group in itertools.groupby(enumerate(jobs),
                                                  lambda t: t[1] - t[0]):
                    group = list(group)
                    if group[0][1] == group[-1][1]:
                        job_intervals.append(str(group[0][1]))
                    else:
                        job_intervals.append(f"{group[0][1]}-{group[-1][1]}")

                log.info(f"{k:<11}: {len(v['jobs']):>2}/{v['max']:<2} running")
                log.info(f"-> ids     : "
                         f"{', '.join(job_intervals)}")

        # delete host which will not receive further jobs
        for td in to_delete:
            self.c.pop(td)

        self._dump2disk()

    def loop(self):
        """Run computational loop."""
        total_jobs = len(self.job_data)
        self.start_time = time()

        while self.job_data:

            self._job_satistics()

            # select server
            for k, v in self.c.items():
                if len(v["jobs"]) < v["max"]:
                    server = k
                    break
            else:
                self._wait()
                continue

            # extract connection so we do not need to write long commands
            c = self.c[server]

            # select appropriate job.
            # jobs are sorted in asccending order based on number of atoms.
            # for aurel select from end and for others from begining
            if server == "aurel":
                data = self.job_data.pop()
            else:
                data = self.job_data.popleft()

            log.info(f"---------- Optimizing {data['index']}: "
                     f"{total_jobs - len(self.job_data)}/{total_jobs} "
                     f"-------------")

            # prepare calculation in loacal dir
            prepare_dir, job_name = self.prepare_calc(data['index'], server,
                                                      data.pop("atoms"))

            submit_dir = c["remote_dir"] / f"{prepare_dir.name}"

            c["conn"].shutil.upload_tree(prepare_dir, submit_dir, quiet=True)

            log.info(f"Scheduling job on {server}")

            # TODO we assume it never fails??
            # run qsub command
            out = c["conn"].subprocess.run(
                [c["submit"], "pbs.job"], suppress_out=True, quiet=True,
                cwd=submit_dir, capture_output=True, encoding="utf-8"
            )

            # record job id if it needs to be killed
            if server == "aurel":
                try:
                    data["job_id"] = f"{JOB_ID.findall(out.stdout)[0]}.0"
                except IndexError:
                    log.warning("Couldn't get job id")
                    data["job_id"] = None
            else:
                data["job_id"] = out.stdout.split(".")[0]

            # record remote dir
            data["running_dir"] = submit_dir

            # record job name
            data["job_name"] = job_name

            # record scan
            data["SCAN"] = self.SCAN

            # put job data in connections dict
            c["jobs"].append(data)

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


def prepare_data() -> List[Atoms]:
    log.warning("reimplement this if different behaviuor is desired")

    log.info("reading gp_iter6_sparse9k.xml.xyz")

    atoms = list(read_xyz("gp_iter6_sparse9k.xml.xyz", index=slice(None)))

    log.info("changing chemical symbols to Ge")
    for i, a in enumerate(atoms):
        species = ["Ge" for _ in range(len(a))]
        atoms[i].set_chemical_symbols(species)

    return atoms


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(name)-8s %(message)s")
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    args = input_parser()

    log.info(f"Running on: {args['remote']}")
    log.info(f"Start:      {args['start']}")
    log.info(f"Stop:       {args['end']}")
    log.info(f"Recompute:  {args['failed_recompute']}")
    log.info(f"SCAN:       {args['SCAN']}\n")

    WORK_DIR = Path.cwd()
    # DATA_DIR = Path(__file__).parent / "data"
    DATA_DIR = WORK_DIR / "data"
    DUMP_FILE = WORK_DIR / "calc_info_persistence.json"
    CVD = 4.7942826905483325e-05  # target kpoint density

    if DUMP_FILE.is_file():
        inpt = input("Dump file present, "
                     "do you want to restart calculation? [y/n]: ")
        if inpt == "y":
            restart = True
        elif inpt == "n":
            restart = False
        else:
            raise ValueError(f"{inpt} answer is not supported, input y/n")
    else:
        restart = False

    if restart:
        r = Recompute.from_json(args['remote'], args["start"], args["end"],
                                recompute_failed=args["failed_recompute"],
                                scan=args["SCAN"], data_dir=DATA_DIR,
                                dump_file=DUMP_FILE, target_KP_density=CVD)
    else:
        SETTINGS = {
            "aurel": {
                "max_jobs": 500,
                "remote_dir": "/gpfs/fastscratch/rynik/recompute/"
            },
            "kohn": {
                "max_jobs": 1,
                "remote_dir": ("/home/rynik/Raid/dizertacka/train_Si/"
                               "recompute_GAP/")
            }
        }
        for h in ("hartree", "fock", "landau", "schrodinger"):
            SETTINGS[h] = SETTINGS["kohn"]

        r = Recompute(args['remote'], args["start"], args["end"],
                      recompute_failed=args["failed_recompute"],
                      scan=args["SCAN"], remote_settings=SETTINGS,
                      work_dir=WORK_DIR, data_dir=DATA_DIR,
                      dump_file=DUMP_FILE, target_KP_density=CVD)

    atoms = prepare_data()
    print(len(atoms))
    r.get_job_data(atoms)
    r.loop()