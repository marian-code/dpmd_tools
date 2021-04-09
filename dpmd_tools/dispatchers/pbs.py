import logging

log = logging.getLogger(__name__)


def batch_script_pbs(
    server: str,
    n_nodes: int,
    ident: str,
    run_this: str,
    priority: bool = True,
    hour_length: int = 12,
) -> str:
    s = ""

    if server in ("kohn", "planck"):
        ppn = 16
    else:
        ppn = 12

    if run_this == "rings":
        ppn = 1

    log.debug(f"setting ppn to {ppn} for server {server}")

    s += "#!/bin/bash\n"
    s += f"#PBS -l nodes={server}:ppn={ppn},walltime={hour_length}:00:00\n"
    s += f"#PBS -q batch\n"
    s += f"#PBS -u rynik\n"
    s += f"#PBS -N {ident}\n"
    s += f"#PBS -e error.txt\n"
    s += f"#PBS -o output.txt\n"
    s += f"#PBS -W umask=0022\n"
    s += f"#PBS -V\n\n"
    s += "cd $PBS_O_WORKDIR\n"

    if run_this.startswith("vasp"):
        s += "NAME=$(echo $(hostname | tr '[:upper:]' '[:lower:]'))\n\n"
        s += 'if [ "$NAME" = "fock" ]\n'
        s += "then\n"
        s += '    VASP="/home/s/Software/vasp.5.4.4_mpi_TS/bin/vasp_std"\n'
        s += "else\n"
        s += (
            '    VASP="/home/s/Software/VASP/intel-mpi-TS-HI/'
            'vasp-5.4.4-TS-HI/bin/vasp_std"\n'
        )
        s += "fi\n\n"
        s += "source /opt/intel/bin/compilervars.sh intel64\n"
        s += 'export PATH="$PATH:/opt/intel/bin"\n'
        s += 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64_lin"\n'
        s += f"/home/s/bin/mpirun -np {ppn} $VASP\n"
    elif run_this == "rings":
        s += f"RINGS=/home/rynik/Software/rings/bin/rings\n"
        # need to input twice y when auto cutoff determination
        #s += f"/home/s/bin/mpirun -np {ppn} printf 'y\\ny\\n' | $RINGS input\n"
        s += f"$RINGS input\n"

    return s
