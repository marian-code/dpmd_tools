import argparse
import atexit
import multiprocessing
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from time import time

from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go
from deap import algorithms, base, creator, tools
from scoop import futures
from loky import get_reusable_executor
from dpmd_tools.utils import init_yappi
from dpmd_tools.represent_2D import _op as Op

try:
    import cupy as np
except ImportError:
    import numpy as np

    CUPY = False
else:
    CUPY = True
    class _cupy_array(np.ndarray):
        def __deepcopy__(self, memo):
            """Overrides the deepcopy from cupy.ndarray that does not copy
            the object's attributes. This one will deepcopy the array and its
            :attr:`__dict__` attribute.
            """
            copy_ = np.ndarray.copy(self)
            copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
            return copy_

        @staticmethod
        def __new__(cls, iterable):
            """Creates a new instance of a cupy.ndarray from a function call.
            Adds the possibility to instanciate from an iterable."""
            return np.array(list(iterable)).view(cls)

        def __setstate__(self, state):
            self.__dict__.update(state)

        def __reduce__(self):
            return (self.__class__, (list(self),), self.__dict__)
finally:
    from numpy import load as np_load

#import numpy as np
#CUPY = False

from dpmd_tools.represent_2D._progress import ProgressBar

WORK_DIR = Path.cwd()
# suppress numpy warning
# RuntimeWarning: invalid value encountered in subtract
warnings.filterwarnings("ignore")


def input_parser():
    p = argparse.ArgumentParser(
        description="Find optimal G2 and G3 sym functions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-G2",
        "--G2-radial",
        action="store_true",
        default=False,
        help="compute radial symmetry function G2",
    )
    p.add_argument(
        "-G3",
        "--G3-angular",
        action="store_true",
        default=False,
        help="compute angular symmetry function G3",
    )
    p.add_argument("-ps", "--popsize", type=int, default=10, help="population size")
    p.add_argument(
        "-g",
        "--generation-number",
        type=int,
        default=15000,
        help="number of EA generations",
    )
    p.add_argument(
        "-mr",
        "--mutation-rate",
        type=float,
        default=0.7,
        help="set mutation probability",
    )
    p.add_argument(
        "-cr",
        "--crossover-rate",
        type=float,
        default=0.3,
        help="set corssover probability",
    )
    p.add_argument(
        "-fn",
        "--function-number",
        type=int,
        default=6,
        help="set the desired number of symmetry functions",
    )
    p.add_argument(
        "-c",
        "--cache",
        action="store_true",
        default=False,
        help="Use cached symfunction values in one file instead of "
        "loading them separately",
    )
    p.add_argument(
        "-p", "--parallel", action="store_true", default=False, help="Run in parallel"
    )
    return vars(p.parse_args())


def plot_evolution(log):

    mins, gens = log.select("min", "gen")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=mins))
    fig.update_layout(
        title="EA fitness function evolution through generations",
        xaxis_title="Generation [N]",
        yaxis_title="Fittness function value [arb. units]",
    )

    fig.write_html(f"energy-generation.html")


class OptimizeProjection:
    def __init__(
        self,
        *,
        popsize: int,
        crossover_rate: float,
        mutation_rate: float,
        number_of_generations: int,
        parallel: bool,
        fingerprint_file: Path,
    ):

        self.POPSIZE = popsize
        self.CROSSOVER_RATE = crossover_rate
        self.MUTATION_RATE = mutation_rate
        self.NUMBER_OF_GEN = number_of_generations

        self.dist_mat_fp = cosine_distances(np_load(fingerprint_file))
        if CUPY:
            self.dist_mat_fp = np.asarray(self.dist_mat_fp)

        self.N_POINTS = self.dist_mat_fp.shape[0]

        self._print_parameters()

        # create toolbox
        self.tb = base.Toolbox()
        #  setup rng
        if CUPY:
            self.rng = np.random
        else:
            self.rng: np.random.Generator = np.random.default_rng()

        # set parallel calculation
        if parallel:
            # pool = get_reusable_executor()
            pool = multiprocessing.Pool()
            self.tb.register("map", pool.map)

        self._init_cache()
        self._init_base()
        self._init_individual()
        self._init_operators()
        self._init_pop()
        self._init_stats()

    def _init_cache(self):

        self.memory = {}
        self.hit = 0
        self.miss = 0

    def _init_base(self):

        # minus for value range, normalized value range and + for correlation
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if CUPY:
            creator.create("Individual", _cupy_array, fitness=creator.FitnessMin)
        else:
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    def _init_individual(self):

        # individual is an array Nx2 with cartesian coordinates of points in 2D
        self.tb.register("attr_int", self.rng.random, (self.N_POINTS, 2))
        self.tb.register(
            "individual", tools.initIterate, creator.Individual, self.tb.attr_int
        )

        print(self.rng.random((5, 5)))
        print(self.tb.attr_int)
        print(self.tb.individual())

    def _init_operators(self):

        # create operators
        self.tb.register(
            "evaluate",
            Op.fitness_function,
            dist_mat_fp=self.dist_mat_fp,
            n_points=self.N_POINTS,
        )
        self.tb.register("mate", Op.cxTwoPointCopy, n_points=self.N_POINTS)
        self.tb.register("mutate", Op.mutCoord, amplitude=0.05)
        self.tb.register("shuffle", Op.shuffleCoord)

        # use selNSGA2 for MultiObjective optimization
        self.tb.register("select", tools.selTournament, tournsize=3)

    def _init_pop(self):

        # create population
        self.tb.register("population", tools.initRepeat, list, self.tb.individual)

        # create initial population and hall of fame
        self.pop = self.tb.population(n=self.POPSIZE)
        print(type(self.pop[0]))
        self.hof = tools.HallOfFame(10, similar=Op.indiv_equal)

    def _init_stats(self):

        # register statistics functions
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)

        self.required_stats = {
            "min": np.min,
            "max": np.max,
            "std": np.std,
            "mean": np.mean,
        }

        for name, function in self.required_stats.items():
            self.stats.register(name, function)

    def cache_info(self):
        print("************ Cache info **************")
        print(f"hits:   {self.hit}")
        print(f"misses: {self.miss}")
        print(f"size:   {len(self.memory)}")
        print("**************************************")

    def run_genetic(self):

        from pprint import pprint

        # pprint(self.__dict__)

        t = time()
        with ProgressBar(self.NUMBER_OF_GEN, self.required_stats.keys()) as p:
            with redirect_stdout(p):
                pop, log = algorithms.eaSimple(
                    self.pop,
                    self.tb,
                    cxpb=self.CROSSOVER_RATE,
                    mutpb=self.MUTATION_RATE,
                    ngen=self.NUMBER_OF_GEN,
                    stats=self.stats,
                    halloffame=self.hof,
                )

        print(f"\nTotal time: {time() - t}\n")
        # self.cache_info()
        # self.fitness_function(self.hof[0], show=True)

        return self.hof, log

    def fitness_function_cache(self, x, show=False):

        x.sort()

        h = hash(x.tostring())

        mem = self.memory.get(h, None)

        if mem:
            self.hit += 1
            return (mem,)
        else:
            self.miss += 1

        # write result into cache
        self.memory[h] = result

        return (result,)

    def _print_parameters(self):

        print(f"Population size is:                        {self.POPSIZE}")
        print(f"Number of structure fingerprints is:       {self.N_POINTS}")
        print(f"Crossover rate is:                         {self.CROSSOVER_RATE}")
        print(f"Mutation rate is:                          {self.MUTATION_RATE}")
        print(f"Number of generations is:                  {self.NUMBER_OF_GEN}")
        print("----------------------------------------")


def write_results(g, hof, log):

    print("Writing results")

    if not hof:
        hof = g.hof

    # plot gen algorithm evolution
    if log:
        print("plotting gen algorithm evolution")
        plot_evolution(log)
    else:
        print("Run was terminated cannot plot evolution")

    points = g.hof[0].T

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=points[0], y=points[1], mode="markers"))
    fig.update_layout(
        title="STRUCTURE distances representation in 2D",
        xaxis_title="x [arb. units]",
        yaxis_title="y [arb. units]",
    )

    fig.write_html(f"2D-representation.html")


if __name__ == "__main__":

    # init
    op = OptimizeProjection(
        popsize=20,
        crossover_rate=0.3,
        mutation_rate=0.7,
        number_of_generations=10000,
        parallel=False,
        fingerprint_file="fingerprints_1.npy",
    )

    # register finalize handler if the script is shut down
    # before it reaches end
    atexit.register(write_results, op, None, None)
    # init_yappi()

    # run EA
    hof, log = op.run_genetic()

    # run was completed unregister exit handler
    atexit.unregister(write_results)

    # write results
    write_results(op, hof, log)

