import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from scipy.spatial import distance_matrix


def genetic_algorithm_benchmark(coords_list):
    fitness_coords = mlrose.TravellingSales(coords=coords_list)
    problem_no_fit = mlrose.TSPOpt(length=20, coords=coords_list,
                                   maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_no_fit, random_state=2)

    print('Genetic algorithm solution is: ', best_state)

    print('Genetic algorithm length is: ', best_fitness)

    return best_state, best_fitness
