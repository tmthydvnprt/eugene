# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
List.py
"""

import copy as cp
import numpy as np

# @profile
def random_list(max_length=0, item_factory=None, eval_function=None):
    """Generate a random list using the item_factory function"""
    return List([item_factory() for _ in xrange(max_length)], item_factory, eval_function)

class List(list):
    """
    Defines an extention of the general list with functions to operate on items specific to eugene.

    items         : the elemnts of the list
    item_factory  : the factory function used to generate new items (assumed randomly), user defined
    eval_function : the function used to convert list into something else, user defined
    """
    def __init__(self, items=None, item_factory=None, eval_function=None):
        """
        """
        super(List, self).__init__(items)
        self.type = 'List'
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
