# pylint: disable=too-many-branches,wildcard-import,unused-wildcard-import
"""
Node.py
"""

import random as r

from eugene.Primatives import *

# @profile
def random_node(max_level=20, min_level=1, current_level=0):
    """
    Create a random node that may contain random subnodes.
    """

    if current_level == max_level:
        rand_node = r.randint(0, 3)
        # node = a constant
        if rand_node == 0:
            node = Node(CONSTS[r.randint(0, len(CONSTS) - 1)])
        # node = EPHEMERAL constant random ( 0:1, uniform -500:500, or normal -500:500 )
        elif rand_node == 1:
            node = Node(EPHEMERAL[r.randint(1, len(EPHEMERAL) - 1)])
        # node = EPHEMERAL constant random integer
        elif rand_node == 2:
            node = Node(EPHEMERAL[0])
        # node = variable
        elif rand_node == 3:
            node = Node(VARIABLES[r.randint(0, len(VARIABLES) - 1)])
    else:
        # rand_node = r.randint(4, 6) if current_level < min_level else r.randint(0, 6)
        rand_node = r.randint(4, 5) if current_level < min_level else r.randint(0, 5)
        # node = a constant
        if rand_node == 0:
            node = Node(CONSTS[r.randint(0, len(CONSTS) - 1)])
        # node = EPHEMERAL constant random ( 0:1, uniform -500:500, or normal -500:500 )
        elif rand_node == 1:
            node = Node(EPHEMERAL[r.randint(1, len(EPHEMERAL) - 1)])
        # node = EPHEMERAL constant random integer
        elif rand_node == 2:
            node = Node(EPHEMERAL[0])
        # node = variable
        elif rand_node == 3:
            node = Node(VARIABLES[r.randint(0, len(VARIABLES) - 1)])
        # node = a unary operator
        elif rand_node == 4:
            node = Node(
                UNARIES[r.randint(0, len(UNARIES) - 1)],
                random_node(max_level, min_level, current_level + 1)
            )
        # node = a binary operator
        elif rand_node == 5:
            node = Node(
                BINARIES[r.randint(0, len(BINARIES) - 1)],
                random_node(max_level, min_level, current_level + 1),
                random_node(max_level, min_level, current_level + 1)
            )
        # # node = a n-ary operator
        # elif rand_node == 6:
        #     nary_node_num = r.randint(2, 5)
        #     if nary_node_num == 2:
        #         node = Node(
        #             NARIES[r.randint(0, len(NARIES) - 1)],
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1)
        #         )
        #     elif nary_node_num == 3:
        #         node = Node(
        #             NARIES[r.randint(0, len(NARIES) - 1)],
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1)
        #         )
        #     elif nary_node_num == 4:
        #         node = Node(
        #             NARIES[r.randint(0, len(NARIES) - 1)],
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1)
        #         )
        #     elif nary_node_num == 5:
        #         node = Node(
        #             NARIES[r.randint(0, len(NARIES) - 1)],
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1),
        #             random_node(max_level - 1, current_level + 1)
        #         )
    return node

class Node(object):
    """
    Defines a node with a value and capability to contain children.
    """

    def __init__(self, value=None, *children):
        # node properties
        self.value = value
        self.children = children
        # current position of node
        self.num = None
        self.level = None
        # summary of children
        self.height = None
        self.node_num = None
        self.leaf_num = None
        self.edge_num = None
        # sum of subtrees
        self.complexity = None

    @property
    def is_leaf(self):
        """
        Check if this node is a leaf.
        """
        return len(self.children) == 0

    @property
    def ary(self):
        """
        Return the arity of the node.
        """
        return len(self.children)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # node is a variable or constant
        if len(self.children) == 0:
            return '%s' % self.value
        # node is a unary, binary or n-ary function
        else:
            if self.value in NARIES:
                return '%s([%s])' % (self.value, ', '.join([str(c) for c in self.children]))
            else:
                return '%s(%s)' % (self.value, ', '.join([str(c) for c in self.children]))

    # @profile
    def set_nums(self, node_counter=-1, level_counter=0, leaf_count=-1, edge_count=-1):
        """
        Set node numbers (depth first).
        """

        # count this node
        node_counter += 1
        self.num = node_counter
        self.level = level_counter
        complexity = 0
        node_count = 1

        # traverse children if present or count as leaf node
        if len(self.children) > 0:
            level_counter += 1
            edge_count += len(self.children)
            height_count = 1
            for c in self.children:
                child_numbers = c.set_nums(node_counter, level_counter, leaf_count, edge_count)
                node_counter, child_node_count, child_height, leaf_count, edge_count, child_complexity = child_numbers
                height_count = max(height_count, child_height)
                complexity += child_complexity
                node_count += child_node_count
        else:
            leaf_count += 1
            height_count = 0
            edge_count = 0

        # store counts of children below
        self.height = height_count
        self.node_num = node_count
        self.leaf_num = leaf_count
        self.edge_num = edge_count
        complexity += node_count
        self.complexity = complexity

        return (node_counter, node_count, height_count + 1, leaf_count, edge_count, complexity)
