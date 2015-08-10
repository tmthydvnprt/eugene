"""
eugene_test.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/Users/timothydavenport/GitHub/eugene')

import eugene.Primatives

eugene.Primatives.x = np.linspace(0, 8.0 * np.pi, 1024)
eugene.Primatives.TRUTH = eugene.Primatives.x * np.sin(eugene.Primatives.x) + eugene.Primatives.x/2.0 + 1.61

from eugene.Population import Population
import numpy as np
import scipy.signal

@profile
def error_and_complexity(gene_expression, scale):
    """user fitness function, weighted combination of error and complexity"""

    weights = np.array([0.95, 0.025, 0.025])
    scaled_gene_expression = 1.0 / (gene_expression / scale)

    return np.dot(scaled_gene_expression, weights)

P = Population(
    init_population_size=1000,
    objective_function=error_and_complexity,
    max_generations=100,
    init_tree_size=2,
    target=eugene.Primatives.TRUTH,
    pruning=False
)

P.initialize()

P.run(50)
