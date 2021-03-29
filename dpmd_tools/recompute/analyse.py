"""Easily batch analyse large amount of structures on remote clusters.

Supports aurel and all our PCs.
Easily restart and pick up in case of fail.
All is done automatically through ssh, just need to supply structures as list
of ase.Atoms objects.
"""

import logging
import re
import signal
from pathlib import Path
from shutil import copy2, rmtree
from typing import TYPE_CHECKING, List, Optional, Tuple

from ase.io import write
from dpmd_tools.dispatchers import Dispatcher

from ._remote_batch_run import RemoteBatchRun

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from dpmd_tools.recompute.json_serializer import Job

CPU_TIME = re.compile(r"\s*Total\s+time\s+for\s+the\s+calculation\s*\|\s*(\S*)\s*\|")

log = logging.getLogger(__name__)


def postprocess_args(args: dict) -> dict:

    if len(args["user"]) == 1:
        args["user"] = args["user"] * len(args["remote"])

    if len(args["max_jobs"]) == 1:
        args["max_jobs"] = args["max_jobs"] * len(args["remote"])

    return args


class Analyse(RemoteBatchRun):
    """Class that encapsulates all the remote server analyse logic."""

    def set_constants(self, *, rings_template: Path, cutoff: float = 0.0):
        """Set constants specific for this subclass.

        Parameters
        ----------
        rings_template : Path
            directory with rings template files
        cutoff : float, optional
            set atoms search cutoff for analysis computations, by default 0.0 - this
            means that cutoff will be determined automatically for each frame from the
            first minimum of RDF
        """
        # set constants
        self.RINGS_TEMPLATE = rings_template
        self.CUTOFF = cutoff

    # * prepare job ********************************************************************
    def prepare_calc(
        self, index: int, calc_dir: Path, server: str, atoms: Atoms
    ) -> Tuple[str, str]:

        log.info("Preparing job on local side...")

        job_name = f"Analyse-{index}"

        # copy rings template files
        copy2(self.RINGS_TEMPLATE / "options", calc_dir)
        (calc_dir / "data").mkdir(exist_ok=True)

        xyz_in = calc_dir / "tmp.xyz"
        xyz_out = calc_dir / "data" / "data.xyz"
        write(xyz_in, atoms, format="xyz")

        # format input data file in the right way, RINGS is extremely picky
        self._xyz_format(xyz_in, xyz_out)
        # mod input file
        self._mod_input(calc_dir, xyz_out, atoms)

        # write identification number and create PBS submit script
        n_nodes = 1 if len(atoms) < 10 else 2
        job_script = Dispatcher.get(server)(
            n_nodes, job_name, "rings", priority=True, hour_length=12
        )

        return job_name, job_script

    @staticmethod
    def _xyz_format(source: Path, output: Path):

        with open(source, "r") as infile, open(output, "w") as outfile:
            lines = infile.readlines()

            for line_num, line in enumerate(lines):

                # first two lines contain number of atoms and comment
                if line_num < 2:
                    outfile.write(line)
                    continue

                symbol, x, y, z = line.split()
                x, y, z = map(float, (x, y, z))

                outfile.write(f"{symbol:<2}  {x:>19.15f}   {y:>19.15f}   {z:>19.15f}\n")

        source.unlink()

    def _mod_input(self, calc_dir: Path, xyz_file: Path, atoms: Atoms):
        """RINGS input file modification."""

        def r(string, to_replace, replace_with):
            spaces = len(to_replace) - len(replace_with)
            if spaces > 0:
                replace_with = replace_with + " " * spaces
            elif spaces < 0:
                string = string.replace(" " * spaces, "")

            return string.replace(to_replace, replace_with)

        symb = list(set(atoms.get_chemical_symbols()))

        with open(self.RINGS_TEMPLATE / "input", "r") as infile, open(
            calc_dir / "input"
        ) as outfile:
            for line in infile:
                if "$na$" in line:
                    line = r(line, "$na$", str(len(atoms)))
                elif "$cell_1$" in line:
                    line = r(
                        line,
                        "$cell_1$",
                        "{} {} {}".format(*atoms.cell[0]),
                    )
                elif "$cell_2$" in line:
                    line = r(
                        line,
                        "$cell_2$",
                        "{} {} {}".format(*atoms.cell[1]),
                    )
                elif "$cell_3$" in line:
                    line = r(
                        line,
                        "$cell_3$",
                        "{} {} {}".format(*atoms.cell[2]),
                    )
                elif "$filename$" in line:
                    line = r(line, "$filename$", xyz_file.name)
                elif "$chem_symbols$" in line:
                    line = r(line, "$chem_symbols$", f"{' '.join(symb)}")
                elif "$n_chem_types$" in line:
                    line = r(line, "$n_chem_types$", str(len(symb)))
                elif "$cutoffs$" in line:
                    c = ""
                    for i, a in enumerate(symb):
                        for b in symb[i:]:
                            c += f"{a:2} {b:2}   {self.CUTOFF}\n"

                    # replace last linebreak
                    c = c.rsplit("\n")[0]
                    line = line.replace("$cutoffs$", c)
                outfile.write(line)

    def postprocess_job(self, job: "Job") -> Optional[float]:
        calc_dir = self.WD / job.running_dir.name

        try:
            calc_time = self._get_cpu_time(calc_dir, "Walltime")
        except (IndexError, FileNotFoundError):
            return None
        else:
            self._cleanup(calc_dir)
            return calc_time

    @staticmethod
    def _cleanup(path: Path):
        """Delete unecessarry dirs and files after job done."""
        DELETE_DIRS = [
            "Gr-p",
            "liste-",
            "gr-bt",
            "gr-p",
            "tmp",
            "al-p",
            "fz-p",
        ]
        DELETE_FILES = [
            "Walltime",
            "con-matrix-",
            "evol-",
            "RINGS-res-",
            "all-al-p.agr",
            "all-al-p.dat",
            "all-fz-p.agr",
            "all-fz-p.dat",
            "sq-neutrons.agr",
            "sq-neutrons.dat",
            "sq-xrays.agr",
            "agr",
        ]

        for d in path.rglob("*"):
            if d.is_dir():
                for dd in DELETE_DIRS:
                    if d.name in dd:
                        try:
                            rmtree(d)
                        except OSError as e:
                            log.debug(f"error in removing directory {d}: {e}")
            elif d.is_file():
                for df in DELETE_FILES:
                    if d.name in df:
                        try:
                            d.unlink()
                        except OSError as e:
                            log.debug(f"error in removing directory {d}: {e}")

    @staticmethod
    def _get_cpu_time(calc_dir, filename: str) -> float:

        output = (calc_dir / filename).read_text()
        return float(CPU_TIME.findall(output)[0])


def prepare_data() -> List[Atoms]:
    log.warning("reimplement this if different behaviuor is desired")

    log.info("reading from XDATCAR files")

    from ase.io import read

    system = read("../XDATCAR", index=slice(None))
    system.extend(read("XDATCAR", index=slice(None)))

    return system


def rings(args):
    args = postprocess_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s: %(message)s",
    )
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    log.info(f"Running on:     {args['remote']}")
    log.info(f"Start:          {args['start']}")
    log.info(f"Stop:           {args['end']}")
    log.info(f"Recompute:      {args['failed_recompute']}")
    log.info(f"Threaded:       {args['threaded']}")
    log.info(f"Usernames:      {args['user']}")
    log.info(f"template dir:   {args['template']}")
    log.info(f"Max queue jobs: {args['max_jobs']}\n")

    WORK_DIR = Path.cwd()
    DUMP_FILE = WORK_DIR / "calc_info_persistence.json"

    if DUMP_FILE.is_file():
        inpt = input("Dump file present, do you want to restart calculation? [y/n]: ")
        if inpt == "y":
            restart = True
        elif inpt == "n":
            restart = False
        else:
            raise ValueError(f"{inpt} answer is not supported, input y/n")
    else:
        restart = False

    if restart:
        r = Analyse.from_json(
            args["remote"],
            args["user"],
            args["start"],
            args["end"],
            recompute_failed=args["failed_recompute"],
            dump_file=DUMP_FILE,
            threaded=args["threaded"],
        )
    else:
        SETTINGS = {}
        for r in args["remote"]:
            user = args["user"][args["remote"].index(r)]
            max_jobs = args["max_jobs"][args["remote"].index(r)]
            if r == "aurel":
                SETTINGS[r] = (
                    {
                        "max_jobs": max_jobs,
                        "remote_dir": f"/gpfs/fastscratch/{user}/analyse/",
                    },
                )
            else:
                SETTINGS[r] = {
                    "max_jobs": max_jobs,
                    "remote_dir": f"/home/{user}/Raid/analyse/",
                }

        r = Analyse(
            args["remote"],
            args["user"],
            args["start"],
            args["end"],
            recompute_failed=args["failed_recompute"],
            remote_settings=SETTINGS,
            work_dir=WORK_DIR,
            dump_file=DUMP_FILE,
            threaded=args["threaded"],
        )

    r.set_constants(rings_template=args["template"], cutoff=3.0)

    atoms = prepare_data()
    log.info(f"got {len(atoms)} structures")
    r.get_job_data(atoms)
    # init_yappi()
    signal.signal(signal.SIGINT, r.handle_ctrl_c)
    r.loop()
