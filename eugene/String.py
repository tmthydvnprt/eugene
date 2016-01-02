# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
String.py
"""

import copy as cp
import numpy as np

# @profile
def random_string(max_length=0, item_factory=None, eval_function=None):
    """Generate a random string using the item_factory function for each character"""
    return String([itemfactory() for _ in xrange(max_length)], item_factory, eval_function)

class String(str):
    """
    Defines an extention of the general string with functions to operate on letters specific to eugene.

    characters    : the elements of the string
    item_factory  : the factory function used to generate new characters (assumed randomly)
    eval_function : the function used to convert the string into something else
    """

    def __init__(self, characters, eval_function=None):
        """
        """
        super(List, self).__init__(items)
        self.type = 'String'
        self.eval_function = eval_function
        self.item_factory = item_factory

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
