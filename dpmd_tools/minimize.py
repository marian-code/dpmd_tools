"""Optimize specified snapshots from MD trajectory."""

import sys
from itertools import islice
from pathlib import Path
from shutil import copy2, rmtree
import subprocess
from threading import Lock

import pandas as pd
import plotly.express as px
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from ase.spacegroup import get_spacegroup
from joblib import Parallel, delayed
from tqdm import tqdm
from os import fspath
import argparse
from time import sleep, time
from ssh_utilities import Connection
from ssh_utilities.exceptions import CalledProcessError

import plotly_theme_setter

sys.path.append("/home/rynik/OneDrive/dizertacka/code/rw")
from lammps import read_lammps_out
from random import choice

WD = Path.cwd()
HOSTS = (
    "kohn",
    "fock",
    "hartree",
    "landau",
    #"schrodinger"
)
TEST = False

if TEST:
    OPT = WD / "opt_data_test"
else:
    OPT = WD / "opt_data"

SYMPREC = 1


def input_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--recompute", default=False, action="store_true",
                   help="if data should be recomputed again")
    p.add_argument("-e", "--every", default=10, type=int,
                   help="only each n-th step from trajectory will be taken, "
                   "has no effect when recompute=False")
    p.add_argument("-p", "--parallel", default=0, type=int,
                   help="compute in parallel, number signifies used CPUs")
    p.add_argument("-t", "--time-step", default=0.002, type=float,
                   help="timestep length in 'ps' used in lmp")
    p.add_argument("-m", "--mtd-step", default=25, type=int,
                   help="number of steps after trajectory was written in lmp")
    p.add_argument("-c", "--continue", default=False, action="store_true",
                   help="continue previous run")
    p.add_argument("-l", "--local", default=False, action="store_true",
                   help="run locally")

    return vars(p.parse_args())


def job_script(host, job_name, proc=1):

    s = "#!/bin/bash\n"
    s += f"#PBS -l nodes={host}:ppn={proc}\n"
    s += "#PBS -l walltime=1:00:00\n"
    # s += "#PBS -l mem=10gb\n"
    s += "#PBS -q batch\n"
    s += "#PBS -u rynik\n"
    s += f"#PBS -N {job_name}\n"
    s += "#PBS -e error.txt\n"
    s += "#PBS -o output.txt\n"
    s += "#PBS -V\n"
    s += "cd $PBS_O_WORKDIR\n"
    s += "source /opt/intel/bin/compilervars.sh intel64\n"
    s += "/home/rynik/Software/anaconda3/bin/lmp -in ge.lammps\n"
    s += "touch done\n"

    return s


def lammps_optimize(atoms, step: float, continue_run: bool, writer=None):

    if writer:
        write = writer
    else:
        write = print

    lmp_dir = OPT / f"step={step}"
    lmp_dir = str(lmp_dir).replace("/mnt/md0", "/home/rynik/Raid")
    lmp_dir = Path(lmp_dir)

    if continue_run and (lmp_dir / "log.lammps").is_file():
        return collect_optimize(lmp_dir, step, writer)

    lmp_dir.mkdir(exist_ok=True)

    write_lammps_data(fspath(lmp_dir / "data.in"), atoms, force_skew=True,
                      atom_style="charge")
    copy2(WD / "ge.lammps", lmp_dir)

    try:
        subprocess.run(["/home/rynik/Software/anaconda3/bin/lmp",
                        "-in", "ge.lammps"],
                       cwd=lmp_dir, check=True, encoding="utf-8",
                       capture_output=True)
    except subprocess.CalledProcessError as e:
        write(str(e.stderr))
        return
    else:
        return collect_optimize(lmp_dir, step, writer)


def lammps_optimize_remote(atoms, step: float, continue_run: bool,
                           conn: dict, writer=None):

    if writer:
        write = writer
    else:
        write = print

    c = conn["c"]
    lock = conn["lock"]

    lmp_dir = OPT / f"step={step}"
    lmp_dir = str(lmp_dir).replace("/mnt/md0", "/home/rynik/Raid")
    lmp_dir = Path(lmp_dir)

    if continue_run and (lmp_dir / "log.lammps").is_file():
        return collect_optimize(lmp_dir, step, writer)

    lmp_dir.mkdir(exist_ok=True)

    write_lammps_data(fspath(lmp_dir / "data.in"), atoms, force_skew=True,
                      atom_style="charge")
    copy2(WD / "ge.lammps", lmp_dir)

    (lmp_dir / "lmp.pbs").write_text(job_script(c.server_name.lower(),
                                                lmp_dir.name))

    try:
        with lock:
            c.shutil.upload_tree(lmp_dir, lmp_dir, remove_after=False,
                                 quiet=True)
            c.subprocess.run(
                ["/opt/pbs/bin/qsub", "lmp.pbs"], suppress_out=True,
                quiet=True, cwd=lmp_dir, check=True, encoding="utf-8",
                capture_output=True)
    except CalledProcessError as e:
        write(str(e.stderr))
        return
    else:
        while True:
            with lock:
                if c.os.isfile(lmp_dir / "done"):
                    break
                else:
                    sleep(5)

        with lock:
            c.shutil.download_tree(lmp_dir, lmp_dir, remove_after=True,
                                   quiet=True)

        if (lmp_dir / "data.out").is_file():
            return collect_optimize(lmp_dir, step, writer)
        else:
            return


def collect_optimize(lmp_dir: Path, step: float, writer=None):

    if writer:
        write = writer
    else:
        write = print

    try:
        atoms_opt = read_lammps_data(fspath(lmp_dir / "data.out"),
                                     style="charge")
    except Exception as e:
        write(str(e))
        return None

    try:
        atoms_spg = get_spacegroup(atoms_opt, symprec=SYMPREC).symbol
    except RuntimeError as e:
        write(str(e))
        atoms_spg = "unknown"

    try:
        en, vol, stress = read_lammps_out(fspath(lmp_dir / "log.lammps"))[:3]
    except Exception as e:
        write(str(e))
        return None
    else:
        stress /= 1000
        en / len(atoms_opt)

    return step, en, vol, stress, atoms_spg


recompute = input_parser()["recompute"]
parallel = input_parser()["parallel"]
every = input_parser()["every"]
time_step = input_parser()["time_step"] * input_parser()["mtd_step"]
continue_run = input_parser()["continue"]
local = input_parser()["local"]

data = []

if recompute:

    if not continue_run:
        try:
            rmtree(OPT)
        except FileNotFoundError:
            pass
        finally:
            OPT.mkdir(exist_ok=True)

    print("reading trajectry")
    t0 = time()

    if TEST:
        f = (WD / "test.traj").open("r")
    else:
        f = (WD / "10gpa_small_bias/trajectory.lammps").open("r")
    
    atoms = list(read_lammps_dump_text(f, index=slice(None)))

    print(f"load time: {(time() - t0):.2f}s")
    print("done")

    job = tqdm(islice(enumerate(atoms, 1), None, None, every),
               ncols=100, total=int(len(atoms) / every))

    if local:
        if parallel:
            exec = delayed(lammps_optimize)
            pool = Parallel(n_jobs=parallel, backend="loky")
            data = pool(exec(a, step * time_step, continue_run, job.write)
                        for step, a in job)
        else:
            for step, a in job:
                data.append(lammps_optimize(a, step * time_step, continue_run,
                                            job.write))
    else:
        WD = Path(str(WD).replace("/mnt/md0", "/home/rynik/Raid"))

        hosts = type('Hosts', (), {})()
        for h in HOSTS:
            setattr(hosts, h, dict(c=Connection.get(h, quiet=True),
                                   lock=Lock()))
            c = getattr(hosts, h)["c"]
            c.mkdir(WD, exist_ok=True)
            c.copy_files(["graph_step-2_num-1.pb"], WD, WD, "put", quiet=True)

        exec = delayed(lammps_optimize_remote)
        pool = Parallel(n_jobs=parallel, backend="threading")
        data = pool(exec(a, step * time_step, continue_run,
                         getattr(hosts, choice(HOSTS)), job.write)
                    for step, a in job)

else:
    print("reading computed data")
    data_dirs = [d for d in OPT.glob("*")]

    job = tqdm(data_dirs, ncols=100, total=len(data_dirs))

    if parallel:
        exec = delayed(collect_optimize)
        pool = Parallel(n_jobs=parallel, backend="loky")
        data = pool(exec(d, float(d.name.split("step=")[1]), job.write)
                    for d in job)
    else:
        for d in job:
            time = float(d.name.split("step=")[1])
            data.append(collect_optimize(d, time, job.write))

    data = [d for d in data if d]

df = pd.DataFrame(data, columns=["time", "energy", "volume", "stress", "spg"])

# directories are uiterated in arbitrary order!
if not recompute:
    df.sort_values(by=["time"], inplace=True)

fig = px.scatter(df, x='time', y="energy", color="spg",
                 title=f"Localy minimized structures, "
                 f"spacegroup symprec={SYMPREC}",
                 labels={"spg", "Space group",
                         "time", "Time [ps]"
                         "energy", "Energy/atom [eV]"},
                 render_mode='webgl')
fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
fig.add_trace(px.line(df, x='time', y="energy", render_mode='webgl').data[0])
fig.write_html("recompute.html", include_plotlyjs="cdn")