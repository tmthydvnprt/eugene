"""
Individual.py
"""

import time
import copy as cp
import numpy as np
import random as r

from eugene.Tree import random_tree
from eugene.List import random_list, List
from eugene.String import random_string, String
from eugene.Primatives import VARIABLES, UNARIES, BINARIES, NARIES, CONSTS, EPHEMERAL

class Individual(object):
    """
    Defines an 'individual' via a set of chromosomes, along with genge expression and mating functions
    """

    def __init__(self, chromosomes=None):
        self.chromosomes = chromosomes
        self.type = chromosomes.type

    @property
    def size(self):
        """return size of individual"""
        return self.chromosomes.node_num if self.type == 'Tree' else len(self.chromosomes)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.chromosomes)

    def display(self):
        """display helper"""
        self.chromosomes.display()

    # @profile
    def compute_gene_expression(self, error_function=None, target=None):
        """compute gene expression by evaluating function stored in tree, and keep track of time"""

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
        """randomly crossover two chromosomes"""

        # create random crossover points
        x1 = r.randint(0, self.size - 1)
        x2 = r.randint(0, spouse.size - 1)

        # clone parent chromosomes
        c1 = cp.deepcopy(self.chromosomes)
        c2 = cp.deepcopy(spouse.chromosomes)

        if self.type == 'Tree':
            # get nodes to cross
            c1n = c1.get_node(x1)
            c2n = c2.get_node(x2)

            # transfer nodes
            if c2n:
                c1.set_node(x1, c2n)
            if c1n:
                c2.set_node(x2, c1n)
        elif self.type == 'List':
            c1 = List(c1[:x1] + c2[x2:], c1.itemfactory, c1.eval_function)
            c2 = List(c2[:x2] + c1[x1:], c2.itemfactory, c2.eval_function)
        elif self.type == 'String':
            pass

        return (Individual(c1), Individual(c2))

    # @profile
    def mutate(self, pruning=False):
        """ alter a random point or node in chromosomes"""

        # randomly select node to mutate
        mpoint = r.randint(0, self.size - 1)

        if self.type == 'Tree':
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

                if node:
                    # constant
                    if node.value in CONSTS:
                        mutated_value = CONSTS[r.randint(0, len(CONSTS) - 1)]
                    # variable
                    elif node.value in VARIABLES:
                        mutated_value = VARIABLES[r.randint(0, len(VARIABLES) - 1)]
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

        elif self.type == 'List':
            self.chromosomes[mpoint] = random_list(1, self.chromosomes.itemfactory, self.chromosomes.eval_function)[0]

        elif self.type == 'String':
            self.chromosomes[mpoint] = random_string(1)
