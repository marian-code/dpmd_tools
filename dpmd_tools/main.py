import argparse
from pathlib import Path

from ssh_utilities import Connection

import dpmd_tools.readers.to_dpdata as readers
from dpmd_tools.cluster import assign_clusters, take_prints
from dpmd_tools.compare_graph import compare_ev
from dpmd_tools.system import MultiSystemsVar
from dpmd_tools.to_deepmd import to_deepmd
from dpmd_tools.scripts import upload
from dpmd_tools.recompute import recompute

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
        "to_deepmd",
        help="Load various data formats to deepmd. Loaded data will be output "
        "to deepmd_data/all - (every read structure) and deepmd_data/for_train - (only "
        "selected structures) dirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_to_deepmd.add_argument(
        "-p",
        "--parser",
        default=None,
        required=True,
        type=str,
        choices=PARSER_CHOICES,
        help="input parser you wish to use",
    )
    parser_to_deepmd.add_argument(
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
    parser_to_deepmd.add_argument(
        "-e", "--every", default=None, type=int, help="take every n-th frame"
    )
    parser_to_deepmd.add_argument(
        "-v",
        "--volume",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures volume. Input as 10.0 31. In [A^3]",
    )
    parser_to_deepmd.add_argument(
        "-n",
        "--energy",
        default=None,
        type=float,
        nargs=2,
        help="constrain structures energy. Input as -5 -2. In [eV]",
    )
    parser_to_deepmd.add_argument(
        "-a",
        "--per-atom",
        default=False,
        action="store_true",
        help="set if energy, energy-dev and volume constraints are "
        "computed per atom or for the whole structure",
    )
    parser_to_deepmd.add_argument(
        "-gp",
        "--get-paths",
        default=None,
        type=str,
        help="if not "
        "specified default function will be used. Otherwise you can input "
        "python code as string that outputs list of 'Path' objects. The "
        "Path object is already imported for you. Example: "
        "-g '[Path.cwd() / \"OUTCAR\"]'",
    )
    parser_to_deepmd.add_argument(
        "-de",
        "--dev-energy",
        default=False,
        type=float,
        nargs=2,
        help="specify energy deviations lower and upper bound for selection",
    )
    parser_to_deepmd.add_argument(
        "-df",
        "--dev-force",
        default=False,
        type=float,
        nargs=2,
        help="specify force deviations lower and upper bound for selection",
    )
    parser_to_deepmd.add_argument(
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
    parser_to_deepmd.add_argument(
        "-m",
        "--mode",
        default="new",
        choices=("new", "append"),
        type=str,
        help="choose data export mode in append "
        "structures will be appended to ones already chosen for "
        "training in previous iteration. In append mode do not specify the "
        "-gp/--get-paths arguments and start the script in deepmd_data dir, "
        "in this mode only dpmd_raw data format is supported",
    )
    parser_to_deepmd.add_argument(
        "-f",
        "--fingerprint-use",
        default=False,
        action="store_true",
        help="if max-select argument is used that this option specifies "
        "that subsample will be selected based on fingerprints",
    )
    parser_to_deepmd.add_argument(
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
    parser_to_deepmd.add_argument(
        "-mf",
        "--min-frames",
        default=30,
        type=int,
        help="specify minimal "
        "munber of frames a system must have. Smaller systems are deleted. "
        "This is due to difficulties in partitioning and inefficiency of "
        "DeepMD when working with such small data",
    )
    parser_to_deepmd.add_argument(
        "-nf",
        "--n-from-cluster",
        default=100,
        type=int,
        help="number of random samples to select from each cluster",
    )
    parser_to_deepmd.add_argument(
        "-cp",
        "--cache-predictions",
        default=False,
        action="store_true",
        help="if true than prediction for current graphs are stored in "
        "running directory so they do not have to be recomputed when "
        "you wish to run the scrip again",
    )
    parser_to_deepmd.add_argument(
        "--save",
        default="input",
        choices=("no", "input", "yes"),
        help="automatically accept when prompted to save changes",
    )
    parser_to_deepmd.add_argument(
        "-b",
        "--block-pbs",
        default=False,
        action="store_true",
        help="put an empty job in PBS queue to stop others from trying to access GPU",
    )
    parser_to_deepmd.add_argument(
        "-dc",
        "--data-collector",
        default="cf",
        choices=COLLECTOR_CHOICES,
        help="choose data collector callable. 'cf' is parallel based on "
        "concurrent.futures and loky",
    )
    parser_to_deepmd.add_argument(
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
    parser_to_deepmd.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="profile this run with yappi",
    )
    parser_to_deepmd.add_argument(
        "-w",
        "--wait_for",
        type=str,
        default=None,
        help="wait for some file to be present typically frozen graph model and only "
        "then start computation. Accepts path to file or dir",
    )

    # * ev *****************************************************************************
    parser_ev = sp.add_parser(
        "ev",
        help="run E-V plots check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_ev.add_argument(
        "-n",
        "--nnp",
        help="input dir with nnp potential, if "
        "none is input then potential in ./nnp_model will be used",
        default=None,
    )
    parser_ev.add_argument(
        "-g",
        "--graph",
        default=None,
        nargs="+",
        help="use deepMD graph(s). Can also input graphs on remote "
        "(server@/path/to/file). Wildcard '*' is also accepted.",
    )
    parser_ev.add_argument(
        "-r",
        "--recompute",
        default=False,
        action="store_true",
        help="if false only collect previous results and don't run lammps",
    )
    parser_ev.add_argument(
        "-e",
        "--equation",
        default="birchmurnaghan",
        type=str,
        choices=("birchmurnaghan", "p3"),
        help="choose equation to fit datapoints",
    )
    parser_ev.add_argument(
        "-t",
        "--train-dir",
        default=None,
        type=str,
        nargs="*",
        help="input directories with data subdirs so data coverage can be computed",
    )
    parser_ev.add_argument(
        "-m",
        "--mtds",
        default=None,
        type=str,
        nargs="*",
        help="input paths to en_vol.npz files from MTD runs, can "
        "be local(e.g. ../run/en_vol.npz) or remote"
        "(e.g. host@/.../en_vol.npz",
    )
    parser_ev.add_argument(
        "-a",
        "--abinit-dir",
        default=None,
        type=str,
        help="path to directory with abiniitio calculations",
    )

    # * take prints ********************************************************************
    parser_prints = sp.add_parser(
        "take-prints",
        help="compute oganov fingerprints for all structures in dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_prints.add_argument(
        "-bs",
        "--batch-size",
        default=1e6,
        type=int,
        help="Size of chunks that will be used to save "
        "fingerprint data, to avoid memory overflow",
    )
    parser_prints.add_argument(
        "-p",
        "--parallel",
        default=False,
        action="store_true",
        help="whether to run fingerprinting in parallel, usually there is speedup "
        "benefit up to an order of magnitude",
    )
    parser_prints.add_argument(
        "-s",
        "--settings-file",
        required=True,
        type=Path,
        help="input file with setting for ase OFP comparator. "
        "https://gitlab.com/askhl/ase/-/blob/master/ase/ga/ofp_comparator.py"
    )

    # * assign clusters to structures **************************************************
    parser_select = sp.add_parser(
        "assign-clusters",
        help="assign cluster number to each structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_select.add_argument(
        "-p",
        "--passes",
        default=5000,
        type=int,
        help="number ov dataset passes of MiniBatch K-means online learning loop.",
    )
    parser_select.add_argument(
        "-bs",
        "--batch_size",
        default=int(1e6),
        type=int,
        help="Data will be iterativelly loaded from files "
        "specified in the fingerprinting phase, and from "
        "those random batches will be chosen of size "
        "specified by this argument",
    )
    parser_select.add_argument(
        "-nc",
        "--n-clusters",
        default=100,
        type=int,
        help="target number of clusters for K-Means algorithm",
    )
    # * upload dataset *****************************************************************
    parser_upload = sp.add_parser(
        "upload",
        help="upload dataset to remote server dir and/or local dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_upload.add_argument(
        "-s",
        "--server",
        type=str,
        choices=Connection.get_available_hosts(),
        help="select target server to upload to",
    )
    parser_upload.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="select target directory to upload to, if not specified, script will "
        "mirror local directory",
    )

    parser_upload.add_argument(
        "-l",
        "--local",
        type=str,
        default=None,
        help="select LOCAL target directory to mirror data uploaded to remote",
    )
    parser_upload.add_argument(
        "-d",
        "--dirs",
        required=True,
        type=str,
        nargs="+",
        help="select data directories. It is assumed that these were prepared by "
        "to_deepmd script and the toplevel directory contains deepmd_data/for_train "
        "dir structure. Accepts also wildcards",
    )

    # * recompute **********************************************************************
    remote_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    remote_parser.add_argument(
        "-s", "--start", help="start of the interval", type=int, default=0
    )
    remote_parser.add_argument(
        "-e", "--end", help="end of the interval (None = end)", type=int, default=None
    )
    remote_parser.add_argument(
        "-r",
        "--remote",
        help="server to run on",
        nargs="+",
        required=True,
        choices=("aurel", "kohn", "schrodinger", "fock", "hartree", "landau", "planck"),
    )
    remote_parser.add_argument(
        "-f",
        "--failed-recompute",
        help="re-run failed jobs",
        action="store_true",
        default=False,
    )
    remote_parser.add_argument(
        "-th",
        "--threaded",
        action="store_true",
        default=False,
        help="run using multithreading. This is only usefull when you have thousands "
        "of very short jobs. Console output order will get messed up",
    )
    remote_parser.add_argument(
        "-u",
        "--user",
        nargs="+",
        type=str,
        required=True,
        help="input either one user name which will be used for all servers or one "
        "for each server in corresponding order",
    )
    remote_parser.add_argument(
        "-m",
        "--max-jobs",
        nargs="+",
        type=int,
        required=True,
        help="set maximum number of jobs in queue for each server. Can be input as a "
        "list with value for each server in corresponding order or as one number that "
        "will be same for all",
    )

    recompute_parser = sp.add_parser(
        "recompute",
        parents=[remote_parser],
        help="script to recompute arbitrarry atoms set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    recompute_parser.add_argument(
        "-S", "--SCAN", help="whether to use SCAN functional", type=bool, default=False
    )

    rings_parser = sp.add_parser(
        "rings",
        parents=[remote_parser],
        help="script to recompute arbitrarry atoms set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rings_parser.add_argument(
        "-t",
        "--template",
        default=Path.cwd() / "data",
        type=Path,
        required=True,
        help="set directory with rings options and input template files"
    )

    args = p.parse_args()
    dict_args = vars(args)

    if args.command == "to_deepmd":
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
        recompute(dict_args)
    elif args.command == None:
        p.print_help()


if __name__ == "__main__":
    main()
