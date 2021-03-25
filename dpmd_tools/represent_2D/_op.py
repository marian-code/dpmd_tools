from typing import Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def cxTwoPointCopy(
    ind1: np.ndarray, ind2: np.ndarray, n_points: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly swaps a subset of array values between two individuals."""

    ind1_original = ind1.copy()
    ind1_original.fitness = ind1.fitness
    ind2_original = ind2.copy()
    ind2_original.fitness = ind2.fitness

    cx1 = np.random.randint(0, n_points - 2)
    cx2 = np.random.randint(cx1, n_points - 1)
    ind1[cx1:cx2], ind2[cx1:cx2] = (
        ind2_original[cx1:cx2],
        ind1_original[cx1:cx2],
    )

    return ind1, ind2


def mutCoord(individual: np.ndarray, amplitude=0.05) -> Tuple[np.ndarray]:
    """Mutate individual coordinates with defined probability."""

    orig_individual = individual.copy()
    orig_individual.fitness = individual.fitness

    individual += np.random.random(individual.shape) * amplitude
    return (individual,)


def shuffleCoord(individual: np.ndarray, fraction: float = 0.2) -> Tuple[np.ndarray]:
    """Shuffle subset of individual coordinates."""

    shuffle_ind = np.random.randint(
        0, individual.shape[0], size=(int(individual.shape[0] * fraction))
    )
    orig_individual = individual.copy()
    orig_individual.fitness = individual.fitness

    np.random.shuffle(individual[shuffle_ind])
    return (individual,)


def fitness_function(
    x: np.ndarray, dist_mat_fp: np.ndarray = np.empty(0), n_points: int = 0
):

    dist_mat_2D = cosine_distances(x)

    result = np.sum(np.abs(dist_mat_2D - dist_mat_fp)) / (n_points ** 2 - n_points)

    return (result,)


def indiv_equal(ind1: np.ndarray, ind2: np.ndarray, atol=0.0001) -> bool:
    return np.allclose(ind1, ind2, atol=atol)