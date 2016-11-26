"""
Individual.py
"""

import time
import copy as cp
import random as r
import numpy as np

import eugene.Config

from eugene.Tree import random_tree
from eugene.Primatives import UNARIES, BINARIES, CONSTS, EPHEMERAL # NARIES,

class Individual(object):
    """
    Defines an 'individual' via a set of chromosomes, along with genge expression and mating functions
    """

    def __init__(self, chromosomes=None):
        self.chromosomes = chromosomes

    @property
    def size(self):
        """
        Return size of individual.
        """
        return self.chromosomes.node_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.chromosomes)

    def display(self, stdout=True):
        """
        Display helper.
        """
        return self.chromosomes.display(stdout=stdout)

    # @profile
    def compute_gene_expression(self, error_function=None, target=None):
        """
        Compute gene expression by evaluating function stored in tree, and keep track of time.
        """

        # evaluate function and time to compute
        t0 = time.time()
        output = self.chromosomes.evaluate()
        t1 = time.time()

        # calculate error of result and time complexity
        error = error_function(output, target)
        time_complexity = t1 - t0
        physical_complexity = self.chromosomes.complexity

        return np.array([error, time_complexity, physical_complexity])

    # @profile
    def crossover(self, spouse=None):
        """
        Randomly crossover two chromosomes.
        """

        # create random crossover points
        x1 = r.randint(0, self.size - 1)
        x2 = r.randint(0, spouse.size - 1)

        # clone parent chromosomes
        c1 = cp.deepcopy(self.chromosomes)
        c2 = cp.deepcopy(spouse.chromosomes)

        # get nodes to cross
        c1n = c1.get_node(x1)
        c2n = c2.get_node(x2)

        # transfer nodes
        if c2n:
            c1.set_node(x1, c2n)
        if c1n:
            c2.set_node(x2, c1n)

        return (Individual(c1), Individual(c2))

    # @profile
    def mutate(self, pruning=False):
        """
        Alter a random node in chromosomes.
        """

        # randomly select node to mutate
        mpoint = r.randint(0, self.size - 1)

        # mutate whole node by replacing children with random subtree
        if r.random() >= 0.5:
            rand_tree = random_tree(2)
            x2 = r.randint(0, rand_tree.node_num - 1)
            node = rand_tree.get_node(x2)
            self.chromosomes.set_node(mpoint, node)
            # check and prune tree with new subtree for inefficiencies
            if pruning:
                self.chromosomes.prune()

        # or just mutate node value based on current type
        else:
            node = self.chromosomes.get_node(mpoint)
            # constant
            if node.value in CONSTS:
                mutated_value = CONSTS[r.randint(0, len(CONSTS) - 1)]
            # variable
            elif node.value in eugene.Config.VAR.keys():
                mutated_value = eugene.Config.VAR.keys()[r.randint(0, len(eugene.Config.VAR.keys()) - 1)]
            # a unary operator
            elif node.value in UNARIES:
                mutated_value = UNARIES[r.randint(0, len(UNARIES) - 1)]
            # a binary operator
            elif node.value in BINARIES:
                mutated_value = BINARIES[r.randint(0, len(BINARIES) - 1)]
            # a n-ary operator
            # elif node.value in NARIES:
            #     mutated_value = NARIES[r.randint(0, len(NARIES) - 1)]
            # EPHEMERAL constant random ( 0:1, uniform -500:500, or normal -500:500 )
            else:
                mutated_value = EPHEMERAL[r.randint(1, len(EPHEMERAL) - 1)]

            # mutate node value (keeps children, if applicable)
            node.value = mutated_value
            self.chromosomes.set_node(mpoint, node)
