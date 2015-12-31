# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
List.py
"""

import copy as cp
import numpy as np

# @profile
def random_list(max_length=0, itemfactory=None):
    """generate a random list"""
    return List([itemfactory() for _ in xrange(max_length)])

class List(list):
    """
    Defines an extention of general list with functions to operate on items specific to eugene
    """
    def __init__(self, items, eval_function=None):
        """
        """
        super(List, self).__init__(items)
        self.type = 'List'
        self.eval_function = eval_function

    @property
    def height(self):
        return 1

    @property
    def edge_num(self):
        return 0

    @property
    def node_num(self):
        return len(self)

    @property
    def leaf_num(self):
        return len(self)

    @property
    def complexity(self):
        return len(self)

    def evaluate(self):
        return self.eval_function(self)
