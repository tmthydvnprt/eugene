# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
String.py
"""

# @profile
def random_string(max_length=0, item_factory=None, eval_function=None, uid=None):
    """
    Generate a random string using the item_factory function for each character.
    """
    return String(''.join([item_factory() for _ in xrange(max_length)]), item_factory=item_factory, eval_function=eval_function, uid=uid)

class String(str):
    """
    Defines an extention of the general string with functions to operate on letters specific to eugene.

    characters    : the elements of the string
    item_factory  : the factory function used to generate new characters (assumed randomly)
    eval_function : the function used to convert the string into something else
    """

    def __new__(cls, characters=None, item_factory=None, eval_function=None, uid=None):
        obj = str.__new__(cls, characters)
        obj.type = 'String'
        obj.item_factory = item_factory
        obj.eval_function = eval_function
        obj.uid = uid
        obj.evaluated = False
        obj.height = 1
        obj.edge_num = 0
        obj._evaluation = None
        return obj

    @property
    def node_num(self):
        """
        Return the number of nodes (length of string).
        """
        return len(self)

    @property
    def leaf_num(self):
        """
        Return the number of leaves (length of string).
        """
        return len(self)

    @property
    def complexity(self):
        """
        Return the complexity of the list (length of string).
        """
        return len(self)

    def evaluate(self):
        """
        Evaluate expression stored in string.
        """
        if not self.evaluated:
            self._evaluation = self.eval_function(self)
            self.evaluated = True
        return self._evaluation
