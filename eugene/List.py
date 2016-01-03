# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
List.py
"""

import copy as cp
import numpy as np

# @profile
def random_list(max_length=0, item_factory=None, eval_function=None, uid=None):
    """Generate a random list using the item_factory function for each element"""
    return List([item_factory() for _ in xrange(max_length)], item_factory, eval_function, uid=uid)

class List(list):
    """
    Defines an extention of the general list with functions to operate on items specific to eugene.

    items         : the elements of the list
    item_factory  : the factory function used to generate new items (assumed randomly)
    eval_function : the function used to convert the list into something else
    """

    def __init__(self, items=None, item_factory=None, eval_function=None, uid=None):
        """
        """
        super(List, self).__init__(items)
        self.type = 'List'
        self.eval_function = eval_function
        self.item_factory = item_factory
        self.uid = uid
        self.evaluated = False

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
        if not self.evaluated:
            self._evaluation = self.eval_function(self)
            self.evaluated = True
        return self._evaluation
