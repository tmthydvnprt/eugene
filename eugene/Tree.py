# pylint: disable=eval-used,bare-except,wildcard-import,unused-wildcard-import
"""
Tree.py
"""

import copy as cp
import numpy as np

from eugene.Primatives import *
from eugene.Node import Node, random_node

# indent branching [level:num] value (height) - {node_num|complexity}
DISPLAY_NODE_STR = '%s%s[%s:%s] %s (%s) - {%s|%s}'

# @profile
def random_tree(max_level=20, min_level=1, current_level=0):
    """generate a random tree of random nodes"""
    return Tree(random_node(max_level, min_level, current_level))

class Tree(object):
    """
    Defines a tree of nodes, with functions to operate on tree and subtrees
    """

    def __init__(self, nodes=None, subtree=False):
        self.nodes = nodes
        if not subtree:
            self.nodes.set_nums()

    @property
    def height(self):
        """
        Return the number of levels in the tree.
        """
        return self.nodes.height + 1

    @property
    def node_num(self):
        """
        Return the number of nodes in the tree.
        """
        return self.nodes.node_num

    @property
    def leaf_num(self):
        """
        Return the number of leaves in the tree.
        """
        return self.nodes.leaf_num + 1

    @property
    def edge_num(self):
        """
        Return the number of edges in the tree.
        """
        return self.nodes.edge_num + 1

    @property
    def complexity(self):
        """
        Return the complexity of the tree (sum of nodes in tree and each subtree).
        """
        return self.nodes.complexity

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.nodes)

    # @profile
    def evaluate(self):
        """evaluate expression stored in tree.
        """
        try:
            result = np.array(eval(compile(self.__str__(), '', 'eval')))
        except:
            result = np.array(np.nan)
        return result

    # @profile
    def get_node(self, n=0):
        """
        Return a node from the tree.
        """

        # search tree until node number is found and take sub tree
        if self.nodes.num == n:
            return cp.deepcopy(self.nodes)
        elif len(self.nodes.children) > 0:
            for c in self.nodes.children:
                cn = Tree(c, subtree=True).get_node(n)
                if cn:
                    return cn
        else:
            return None

    # @profile
    def set_node(self, n=0, node=None):
        """
        Set a node in the tree.
        """

        # search tree until node number is found, and store sub tree
        if self.nodes.num == n:
            self.nodes = node
        else:
            self.nodes.children = tuple([Tree(c, subtree=True).set_node(n, node) for c in self.nodes.children])

        # rebase the numbers of the Tree
        self.nodes.set_nums()
        return self.nodes

    # @profile
    def list_edges(self):
        """
        Get edges of tree.
        """

        # get list of tuple edges between nodes e.g. [(n1,n2),(n1,n3)...]
        edges = [(self.nodes.value, c.value if len(c.children) > 0 else c.value) for c in self.nodes.children]
        children_nodes = [Tree(c, subtree=True).list_edges() for c in self.nodes.children if len(c.children) > 0]
        for i in xrange(len(children_nodes)):
            edges += children_nodes[i]
        return edges

    # @profile
    def list_nodes(self):
        """
        Return nodes of tree.
        """

        # get list of nodes
        node_list = []
        node_list.append(self.nodes.value)
        # add children
        node_list.extend([c.value for c in self.nodes.children if len(c.children) == 0])
        # add children's children
        grand_children = [Tree(c, subtree=True).list_nodes() for c in self.nodes.children if len(c.children) > 0]
        node_list.extend([node for grand_child in grand_children for node in grand_child])

        return node_list

    # @profile
    def prune(self):
        """
        Go thru nodes and remove or replace dead / constant branches (subtrees).
        """

        # create subtree
        sub_tree = Tree(self.nodes, subtree=True)
        # check if the tree contains a variable
        contains_variable = any([n in VARIABLES for n in sub_tree.list_nodes()])
        # evaluate subtree for inefficiencies
        sub_eval = sub_tree.evaluate()
        # check is evaluation exactly equals one of the variables
        equals_variable = [v for v in VARIABLES if np.array(Tree(Node(v)).evaluate() == sub_eval).all()]

        # if subtree of node does not contain variable, it must be constant
        if not contains_variable:
            self.nodes.value = sub_eval
            self.nodes.children = ()
        # if subtree contains a variable, but evaluates to exactly the variable, replace with variable
        elif equals_variable:
            self.nodes.value = equals_variable[0]
            self.nodes.children = ()
        # can't make more effiecient
        else:
            for child in self.nodes.children:
                Tree(child, subtree=True).prune()

        # rebase the numbers of the Tree
        self.nodes.set_nums()
        return self.nodes

    def display(self, level=0, level_list=None):
        """
        Display helper.
        """
        level_list = level_list if level_list else []

        if level == 0:
            node_str = DISPLAY_NODE_STR % (
                '',
                '',
                self.nodes.level,
                self.nodes.num,
                self.nodes.value,
                self.nodes.height,
                self.nodes.node_num,
                self.nodes.complexity
            )
        else:
            branching = '\\-' if level_list[-1] == '      ' else '|-'
            indent = '    ' + ''.join(level_list[:-1])
            node_str = DISPLAY_NODE_STR % (
                indent,
                branching,
                self.nodes.level,
                self.nodes.num,
                self.nodes.value,
                self.nodes.height,
                self.nodes.node_num,
                self.nodes.complexity
            )
        print node_str
        for i, child in enumerate(self.nodes.children):
            Tree(child, subtree=True).display(level + 1, level_list + ['      ' if i == len(self.nodes.children) - 1 else '|     '])
