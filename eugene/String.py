# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
String.py
"""

import copy as cp
import numpy as np

# @profile
def random_string(max_length=0, itemfactory=None):
    """generate a random list"""
    return String([itemfactory() for _ in xrange(max_length)])

class String(str):
    """
    Defines an extention of general string with functions to operate on letters specific to eugene
    """
    def __init__(self, items, eval_function=None):
        """
        """
        super(List, self).__init__(items)
        self.type = 'String'
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
