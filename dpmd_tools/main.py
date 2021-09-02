import argparse
from pathlib import Path

from ssh_utilities import Connection

import dpmd_tools.readers.to_dpdata as readers
from dpmd_tools.cluster import assign_clusters, take_prints
from dpmd_tools.compare_graph import compare_ev
from dpmd_tools.recompute import recompute, rings
from dpmd_tools.scripts import analyse_mtd, copy_train, run_singlepoint, upload, dev2ase
from dpmd_tools.system import MultiSystemsVar
from dpmd_tools.to_deepmd import to_deepmd

PARSER_CHOICES = [r.replace("read_", "") for r in readers.__all__]
COLLECTOR_CHOICES = [
    v.replace("collect_", "") for v in vars(MultiSystemsVar) if "collect_" in v
]


def main():

    p = argparse.ArgumentParser(
        description="dpmd-tools an enhanced data manager for deepmd",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp = p.add_subparsers(title="Valid subcommands", dest="command")

    # * to_deepmd **********************************************************************
    parser_to_deepmd = sp.add_parser(
        "to-deepmd",
        help="Load various data formats to deepmd. Loaded data will be output "
        "to deepmd_data/all - (every read structure) and deepmd_data/for_train - (only "
        "selected structures) dirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    to_deepmd_parser(parser_to_deepmd)

    # * ev *****************************************************************************
    parser_ev = sp.add_parser(
        "ev",
        help="run E-V plots check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ev_parser(parser_ev)

    # * take prints ********************************************************************
    parser_prints = sp.add_parser(
        "take-prints",
        help="compute oganov fingerprints for all structures in dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prints_parser(parser_prints)

    # * assign clusters to structures **************************************************
    parser_select = sp.add_parser(
        "assign-clusters",
        help="assign cluster number to each structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    assign_clusters_parser(parser_select)

    # * upload dataset *****************************************************************
    parser_upload = sp.add_parser(
        "upload",
        help="upload dataset to remote server dir and/or local dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    upload_dataset_parser(parser_upload)

    # * remote parser base**************************************************************
    remote_parser = get_remote_parser()

    # * recompute VASP *****************************************************************
    recompute_parser = sp.add_parser(
        "recompute",
        parents=[remote_parser],
        help="script to recompute arbitrarry structures set with VASP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remote_recompute_parser(recompute_parser)

    # * analyse rings ******************************************************************
    rings_parser = sp.add_parser(
        "rings",
        parents=[remote_parser],
        help="script to analyse arbitrarry structures set with R.I.N.G.S",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remote_analyse_parser(rings_parser)

    # * run remote VASP ***************************************************************
    run_singlepoint_parser = sp.add_parser(
        "run-singlepoint",
        help="script to run VASP/QE simulation remotely, "
        "suitable only for short one-off jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    singlepoint_parser(run_singlepoint_parser)

    # * analyse-mtd ********************************************************************
    analyse_mtd_p = sp.add_parser(
        "analyse-mtd",
        help="upload dataset to remote server dir and/or local dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyse_mtd_parser(analyse_mtd_p)

    # * copy-train ********************************************************************
    copy_train_p = sp.add_parser(
        "copy-train",
        help="upload dataset to remote server dir and/or local dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    copy_train_parser(copy_train_p)

    # * traj_dev2ase ********************************************************************
    traj_dev2ase = sp.add_parser(
        "dev2ase",
        help="select trajectory frames from lammps run to recompute based on DeepMD "
        "calculated deviations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dev2ase_parser(traj_dev2ase)

    args = p.parse_args()
    dict_args = vars(args)

    if args.command == "to-deepmd":
        to_deepmd(dict_args)
    elif args.command == "ev":
        compare_ev(dict_args)
    elif args.command == "take-prints":
        take_prints(dict_args)
    elif args.command == "assign-clusters":
        assign_clusters(dict_args)
    elif args.command == "upload":
        upload(dict_args)
    elif args.command == "recompute":
        recompute(dict_args)
    elif args.command == "rings":
        rings(dict_args)
    elif args.command == "run-singlepoint":
        run_singlepoint(**dict_args)
    elif args.command == "analyse-mtd":
        analyse_mtd(**dict_args)
    elif args.command == "copy-train":
        copy_train(**dict_args)
    elif args.command == "dev2ase":
        dev2ase(**dict_args)
    elif args.command == None:
        p.print_help()
    else:
        print("Wrong command choice!")
        p.print_help()


def to_deepmd_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-p",
        "--parser",
        default=None,
        required=True,
        type=str,
        choices=PARSER_CHOICES,
        help="input parser you wish to use",
    )
    parser.add_argument(
        "-ts",
        "--take-slice",
        nargs="+",
        type=int,
        default=[None],
        help="some file formats may contain more than one frame in file, like "
        "OUTCAR for instance with this option, you can control which frames "
        "will be taken. e.g. --take-slice 5 -5. This will work as python "
        "list slicing frames = list(frames)[5:-5]",
    )
    parser.add_argument(
        "-g",
        "--graphs",
        default=[],
        type=str,
        nargs="*",
        help="input list of graphs you wish to use for checking if "
        "datapoint is covered by current model. If present must "
        "have at least two distinct graphs. You can use glob patterns "
        "relative to current path e.g. '../ge_all_s1[3-6].pb'. Files can "
        "also be located on remote e.g. "
        "'kohn@'/path/to/file/ge_all_s1[3-6].pb'",
    )
    parser.add_argument(
        "-e", "--every", default=None, type=int, help="take every n-th frame"
    )
    parser.add_argument(
        "-v",
        "--volume",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures volume. Input as 10.0 31. In [A^3]",
    )
    parser.add_argument(
        "-n",
        "--energy",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures energy. Input as -5 -2. In [eV]",
    )
    parser.add_argument(
        "-pr",
        "--pressure",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures pressure. Input as 1.0 50. In GPa",
    )
    parser.add_argument(
        "-a",
        "--per-atom",
        default=False,
        action="store_true",
        help="set if energy, energy-dev and volume constraints are "
        "computed per atom or for the whole structure",
    )
    parser.add_argument(
        "-gp",
        "--get-paths",
        default=None,
        type=str,
        help="if not "
        "specified default function will be used. Otherwise you can input "
        "python code as string that outputs list/generator of 'Path' objects. The "
        "Path object is already imported for you. Example: "
        "-gp '[Path.cwd() / \"OUTCAR\"]'",
    )
    parser.add_argument(
        "-de",
        "--dev-energy",
        default=False,
        type=float,
        nargs=2,
        help="specify energy deviations lower and upper bound for selection",
    )
    parser.add_argument(
        "-df",
        "--dev-force",
        default=False,
        type=float,
        nargs=2,
        help="specify force deviations lower and upper bound for selection",
    )
    parser.add_argument(
        "--std-method",
        default=False,
        action="store_true",
        help="method to use in forces and energy error estimation. Default=False means "
        "that root mean squared prediction error will be used, this will output high "
        "error even if all models predictions aggre but have a constant shift from DFT "
        "data. If true than insted standard deviation in set of predictions by "
        "different models will be used, this will not account for any prediction "
        "biases.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="new",
        choices=("new", "append", "merge"),
        type=str,
        help="choose data export mode. In append mode structures will be appended to "
        "ones already chosen for training in previous iteration. In append mode do not "
        "specify the -gp/--get-paths arguments and start the script in deepmd_data "
        "dir, in this mode only dpmd_raw data format is supported. In merge mode "
        "use -gp/--get-paths argument to specify all directories at once. You must "
        "also specify -md/--merge-dir argument. In this mode no selection criteria are "
        "applied, systems are only read in, merged and output to specified dir",
    )
    parser.add_argument(
        "-md",
        "--merge-dir",
        default=Path.cwd(),
        type=Path,
        help="target dir where read in systems will be merged",
    )
    parser.add_argument(
        "-f",
        "--fingerprint-use",
        default=False,
        action="store_true",
        help="if max-select argument is used that this option specifies "
        "that subsample will be selected based on fingerprints",
    )
    parser.add_argument(
        "-ms",
        "--max-select",
        default=None,
        type=str,
        help="set max number "
        "of structures that will be selected. If above conditions produce "
        "more, subsample will be selected randomly, or based on "
        "fingerprints if available. Can be also input as a percent of all "
        "dataset e.g. 10%% from 5000 = 500 frams selected. The percent "
        "option computes the potrion from whole dataset length not only "
        "from unselected structures",
    )
    parser.add_argument(
        "-mf",
        "--min-frames",
        default=30,
        type=int,
        help="specify minimal "
        "munber of frames a system must have. Smaller systems are deleted. "
        "This is due to difficulties in partitioning and inefficiency of "
        "DeepMD when working with such small data",
    )
    parser.add_argument(
        "-nf",
        "--n-from-cluster",
        default=100,
        type=int,
        help="number of random samples to select from each cluster",
    )
    parser.add_argument(
        "-cp",
        "--cache-predictions",
        default=False,
        action="store_true",
        help="if true than prediction for current graphs are stored in "
        "running directory so they do not have to be recomputed when "
        "you wish to run the scrip again",
    )
    parser.add_argument(
        "--save",
        default="input",
        choices=("no", "input", "yes"),
        help="automatically accept when prompted to save changes",
    )
    parser.add_argument(
        "-b",
        "--block-pbs",
        default=False,
        action="store_true",
        help="put an empty job in PBS queue to stop others from trying to access GPU",
    )
    parser.add_argument(
        "-dc",
        "--data-collector",
        default="cf",
        choices=COLLECTOR_CHOICES,
        help="choose data collector callable. 'cf' is parallel based on "
        "concurrent.futures and loky. Use debug loader if loading fails for some "
        "reason and you want to see the the whole error tracebacks",
    )
    parser.add_argument(
        "-fi",
        "--force-iteration",
        type=int,
        default=None,
        help="When selecting force to use supplied iteration as default, instead of "
        "using the last one. This is usefull when you have some unsatisfactory "
        "iterations and want to revert the selection to some previous one. E.g. when "
        "you have 4 selection iterations the next will be 5-th and will build on data "
        "selected in previous 4. But if '-fi 2' you will build on data selected only "
        "in previous 2. You can also input negative number e.g. -2 which will have the "
        "same effect in this case giving you the 2. generation as base",
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="profile this run with yappi",
    )
    parser.add_argument(
        "-w",
        "--wait-for",
        type=str,
        nargs="*",
        default=None,
        help="wait for some file(s) to be present typically frozen graph model(s) and "
        "only then start computation. Accepts path to file or dir, if argument is "
        "'graphs' then files from --graphs argument are used. You can also input more "
        "paths or use wildcards and shell patterns",
    )


def ev_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-n",
        "--nnp",
        help="input dir with nnp potential, if "
        "none is input then potential in ./nnp_model will be used",
        default=None,
    )
    parser.add_argument(
        "-g",
        "--graph",
        default=None,
        nargs="+",
        help="use deepMD graph(s). Can also input graphs on remote "
        "(server@/path/to/file). Wildcard '*' is also accepted.",
    )
    parser.add_argument(
        "-r",
        "--recompute",
        default=False,
        action="store_true",
        help="if false only collect previous results and don't run lammps",
    )
    parser.add_argument(
        "-e",
        "--equation",
        default="birchmurnaghan",
        type=str,
        choices=("birchmurnaghan", "p3"),
        help="choose equation to fit datapoints",
    )
    parser.add_argument(
        "-t",
        "--train-dir",
        default=None,
        type=str,
        nargs="*",
        help="input directories with data subdirs so data coverage can be computed",
    )
    parser.add_argument(
        "-m",
        "--mtds",
        default=None,
        type=str,
        nargs="*",
        help="input paths to en_vol.npz files from MTD runs, can "
        "be local(e.g. ../run/en_vol.npz) or remote"
        "(e.g. host@/.../en_vol.npz",
    )
    parser.add_argument(
        "-a",
        "--abinit-dir",
        default=None,
        type=str,
        help="path to directory with abiniitio calculations",
    )
    parser.add_argument(
        "-s",
        "--reference-structure",
        default="cd",
        type=str,
        help="choose reference structure for H(p) plot",
    )
    parser.add_argument(
        "-se",
        "--shift-ev",
        default=False,
        action="store_true",
        help="Shift all E(V) curves according to the lowest point E0, V0 point",
    )
    parser.add_argument(
        "-xh",
        "--x-span-hp",
        default=(-10, 40),
        nargs=2,
        type=float,
        help="set the span of x axis in the H(p) graph, applies only to png"
    )
    parser.add_argument(
        "-yh",
        "--y-span-hp",
        default=(-1.5, 1),
        nargs=2,
        type=float,
        help="set the span of y axis in the H(p) graph, applies only to png"
    )


def prints_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-bs",
        "--batch-size",
        default=1e6,
        type=int,
        help="Size of chunks that will be used to save "
        "fingerprint data, to avoid memory overflow",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        default=False,
        action="store_true",
        help="whether to run fingerprinting in parallel, usually there is speedup "
        "benefit up to an order of magnitude",
    )
    parser.add_argument(
        "-s",
        "--settings-file",
        required=True,
        type=Path,
        help="input file with setting for ase OFP comparator. "
        "https://gitlab.com/askhl/ase/-/blob/master/ase/ga/ofp_comparator.py",
    )


def assign_clusters_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-p",
        "--passes",
        default=5000,
        type=int,
        help="number ov dataset passes of MiniBatch K-means online learning loop.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=int(1e6),
        type=int,
        help="Data will be iterativelly loaded from files "
        "specified in the fingerprinting phase, and from "
        "those random batches will be chosen of size "
        "specified by this argument",
    )
    parser.add_argument(
        "-nc",
        "--n-clusters",
        default=100,
        type=int,
        help="target number of clusters for K-Means algorithm",
    )


def upload_dataset_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-s",
        "--server",
        type=str,
        choices=Connection.get_available_hosts(),
        help="select target server to upload to",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="select target directory to upload to, if not specified, script will "
        "mirror local directory",
    )

    parser.add_argument(
        "-l",
        "--local",
        type=str,
        default=None,
        help="select LOCAL target directory to mirror data uploaded to remote",
    )
    parser.add_argument(
        "-d",
        "--dirs",
        required=True,
        type=str,
        nargs="+",
        help="select data directories. It is assumed that these were prepared by "
        "to_deepmd script and the toplevel directory contains deepmd_data/for_train "
        "dir structure. Accepts also wildcards",
    )


def remote_recompute_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-S", "--SCAN", help="whether to use SCAN functional", type=bool, default=False
    )
    parser.add_argument(
        "-l",
        "--loader",
        help="input <dir>.<file_ithout_ext>.<python function>. This must contain function that will be "
        "imported and run. Function must not take any arguments and must return list "
        "of atoms objects to recompute",
        type=str,
        required=True,
    )


def remote_analyse_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-t",
        "--template",
        default=Path.cwd() / "data",
        type=Path,
        required=True,
        help="set directory with rings options and input template files",
    )


def singlepoint_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-s",
        "--server",
        help="remote server to run on",
        type=str,
        required=True,
        choices=Connection.get_available_hosts(),
    )
    parser.add_argument(
        "-l",
        "--local",
        type=Path,
        default=Path.cwd(),
        help="local directory with VASP files",
    )
    parser.add_argument(
        "-r",
        "--remote",
        type=Path,
        default=Path.cwd(),
        help="remote directory to run VASP simulation",
    )
    parser.add_argument(
        "-w",
        "--what",
        type=str,
        choices=("VASP", "QE"),
        default="VASP",
        help="choose what code to run",
    )


def get_remote_parser():

    p = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("-s", "--start", help="start of the interval", type=int, default=0)
    p.add_argument(
        "-e", "--end", help="end of the interval (None = end)", type=int, default=None
    )
    p.add_argument(
        "-r",
        "--remote",
        help="server to run on",
        nargs="+",
        required=True,
        choices=(
            "aurel",
            "kohn",
            "schrodinger",
            "fock",
            "hartree",
            "landau",
            "planck",
            "local",
        ),
    )
    p.add_argument(
        "-f",
        "--failed-recompute",
        help="re-run failed jobs",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "-th",
        "--threaded",
        action="store_true",
        default=False,
        help="run using multithreading. This is only usefull when you have thousands "
        "of very short jobs. Console output order will get messed up",
    )
    p.add_argument(
        "-u",
        "--user",
        nargs="+",
        type=str,
        required=True,
        help="input either one user name which will be used for all servers or one "
        "for each server in corresponding order",
    )
    p.add_argument(
        "-m",
        "--max-jobs",
        nargs="+",
        type=int,
        required=True,
        help="set maximum number of jobs in queue for each server. Can be input as a "
        "list with value for each server in corresponding order or as one number that "
        "will be same for all",
    )

    return p


def analyse_mtd_parser(parser):

    parser.add_argument(
        "-ev",
        "--ev-only",
        help="output only ev file",
        default=False,
        action="store_true",
    )


def copy_train_parser(parser):

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="choose directory from which will new train dir be created",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="choose destination directory",
    )
    parser.add_argument(
        "-c", "--control", type=str, required=True, help="choose training control file",
    )


def dev2ase_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-t",
        "--trajectory-file",
        type=Path,
        default=Path("trajectory.lammps"),
        help="input lammps trajectory file",
    )
    parser.add_argument(
        "-de",
        "--dev-energy",
        default=False,
        type=float,
        nargs=2,
        help="specify energy deviations lower and upper bound for selection",
    )
    parser.add_argument(
        "-df",
        "--dev-force",
        default=False,
        type=float,
        nargs=2,
        help="specify force deviations lower and upper bound for selection",
    )
    parser.add_argument(
        "-p",
        "--portion",
        type=float,
        default=None,
        help="specify portion of the selected frames to take "
        "from trajectory to recompute. Must specify this or nframes",
    )
    parser.add_argument(
        "-n",
        "--nframes",
        type=float,
        default=None,
        help="specify n of the selected frames to take "
        "from trajectory to recompute. Must specify this or portion.",
    )
    parser.add_argument(
        "-li",
        "--lammps-infile",
        type=str,
        default="in.lammps",
        help="name of the lammps input file, relative to parent dir of "
        "lammps trajectory file"
    )
    parser.add_argument(
        "-pi",
        "--plumed-infile",
        type=str,
        default="plumed.dat",
        help="name of the plumed input file, relative to parent dir of "
        "lammps trajectory file"
    )


if __name__ == "__main__":
    main()
