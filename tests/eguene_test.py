"""
eugene_test.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.expanduser('~/GitHub/eugene'))

import eugene.Config
from eugene.Population import Population

# Setup up variable and truth configuration
eugene.Config.VAR['x'] = np.linspace(0, 8.0 * np.pi, 1024)
eugene.Config.TRUTH = eugene.Config.VAR['x'] * np.sin(eugene.Config.VAR['x']) + eugene.Config.VAR['x']/2.0 + 1.61

# @profile
def error_and_complexity(gene_expression, scale):
    """user fitness function, weighted combination of error and complexity"""

    weights = np.array([0.95, 0.025, 0.025])
    scaled_gene_expression = 1.0 / (gene_expression / scale)

    return np.dot(scaled_gene_expression, weights)

# Setup Population
P = Population(
    init_population_size=1000,
    objective_function=error_and_complexity,
    max_generations=100,
    init_tree_size=2,
    target=eugene.Config.TRUTH,
    pruning=False
)

# Initialize Population
P.initialize()

# Run the Population
P.run(20)
