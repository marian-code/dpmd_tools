import logging

log = logging.getLogger(__name__)


def local_run_script(
    server: str,
    n_nodes: int,
    ident: str,
    run_this: str,
    steps: int = 1,
    priority: bool = True,
    hour_length: int = 12,
) -> str:
    s = ""

    ppn = 4

    log.debug(f"setting ppn to {ppn} for server {server}")

    s += "#!/bin/bash\n"

    if run_this.startswith("vasp"):
        s += 'VASP="/home/rynik/Software/VASP/intel-mpi-TS-HI/vasp-5.4.4-TS-HI/bin/vasp_std"\n'
        s += "source /opt/intel/bin/compilervars.sh intel64\n"
        s += 'export PATH="$PATH:/opt/intel/bin"\n'
        s += 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64_lin"\n'
        
        s += f"for ((i=1;i<={steps};i++))\n"
        s += f"do\n"
        s += f"    cp INCAR.$i INCAR\n"
        s += f"    /home/s/bin/mpirun -np {ppn} $VASP\n"
        s += f"    cp CONTCAR POSCAR\n"
        s += f"    cp CONTCAR CONTCAR.$i\n"
        s += f"    cp OSZICAR OSZICAR.$i\n"
        s += f"    cp OUTCAR OUTCAR.$i\n"
        s += f"    cp XDATCAR XDATCAR.$i\n"
        s += f"done\n"
    elif run_this == "rings":
        s += f"RINGS=/home/rynik/Software/rings-v1.3.2/bin/rings\n"
        # need to input twice y when auto cutoff determination
        #s += f"/home/s/bin/mpirun -np {ppn} printf 'y\\ny\\n' | $RINGS input\n"
        s += f"/home/s/bin/mpirun -np {ppn} $RINGS input\n"

    return s
