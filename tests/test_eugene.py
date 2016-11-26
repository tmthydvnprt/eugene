"""
Eguene.py - tests the eugene package.
"""

import unittest

import numpy as np

from eugene.Primatives import *

import eugene.Util as util
import eugene.Tree as tree
import eugene.Node as node

# Test cases
class UtilTests(unittest.TestCase):
    """
    Test the utility module.
    """

    def setUp(self):
        """Setup Tests"""

        self.predicted = np.array([6.0, 10.0, 14.0, 16.0, 7.0, 5.0, 5.0, 13.0, 12.0, 13.0, 8.0, 5.0])
        self.truth = np.array([7.0, 10.0, 12.0, 10.0, 10.0, 8.0, 7.0, 8.0, 11.0, 13.0, 10.0, 8.0])
        self.error = np.sqrt(np.array([1.0, 0.0, 4.0, 36.0, 9.0, 9.0, 4.0, 25.0, 1.0, 0.0, 4.0, 9.0]).mean())

    def test_rmse(self):
        """Test that RMSE is calculated"""

        error = util.rmse(self.predicted, self.truth)
        self.assertEqual(self.error, error)

    def test_rmse_shape(self):
        """Test that RMSE fails when shapes do not match"""

        error = util.rmse(self.predicted[:-2], self.truth)
        self.assertEqual(np.inf, error)

    def test_rmse_nan(self):
        """Test that RMSE fails when NaN is present"""

        predicted = self.predicted
        predicted[4] = np.NaN
        error = util.rmse(predicted, self.truth)
        self.assertEqual(np.inf, error)

    def test_rmse_datatype(self):
        """Test that RMSE fails when wrong data type is passed"""

        error = util.rmse(list(self.predicted), self.truth)
        self.assertEqual(np.inf, error)

    def test_progress_bar(self):
        """Run the progress bar"""

        p = util.ProgressBar(len(self.predicted))

        for i in xrange(len(self.predicted)):
            p.animate(i)

class RandomNodeTests(unittest.TestCase):
    """
    Test creation of Random Nodes.
    """

    def setUp(self):
        """Setup Tests"""
        pass

    def test_random_trees(self):
        """Generate 500 random trees"""
        NUM_NODES = 500
        random_nodes = [node.random_node(10) for i in xrange(NUM_NODES)]

        self.assertEqual(len(random_nodes), NUM_NODES)

class NodeTests(unittest.TestCase):
    """
    Test Node Module.
    """

    def setUp(self):
        """Setup Tests"""
        self.variable_node = node.Node('x')
        self.constant_node = node.Node(3)
        self.ephemeral_node = node.Node(EPHEMERAL[0])
        self.unary_node = node.Node(n_abs, node.Node(-3))
        self.binary_node = node.Node(n_add, node.Node(-3), node.Node(10))
        self.x = np.arange(0, 5)
        self.TRUTH = self.x + (10 * abs(-4))
        self.node = node.Node(n_add,
            node.Node('x'),
            node.Node(n_mul,
                node.Node(10),
                node.Node(n_abs,
                    node.Node(-4)
                )
            )
        )

        self.variable_node.set_nums()
        self.constant_node.set_nums()
        self.ephemeral_node.set_nums()
        self.unary_node.set_nums()
        self.binary_node.set_nums()
        self.node.set_nums()

    def test_node_numbers(self):
        """Check node numbers"""
        node_nums = self.node.set_nums()
        self.assertEqual(node_nums, (5, 6, 4, 3, 5, 15))

    def test_node_string(self):
        """Check node string printing"""
        correct_str = '<built-in function add>(x, <built-in function mul>(10, <built-in function abs>(-4)))'
        node_str = self.node.__repr__()
        self.assertEqual(node_str, correct_str)

    def test_node_arity(self):
        """Check node Arity"""

        nodes = [
            self.variable_node,
            self.constant_node,
            self.ephemeral_node,
            self.unary_node,
            self.binary_node,
            self.node
        ]
        arities = [n.ary for n in nodes]
        self.assertEqual(arities, [0, 0, 0, 1, 2, 2])

    def test_node_leafiness(self):
        """Check node leafiness"""

        nodes = [
            self.variable_node,
            self.constant_node,
            self.ephemeral_node,
            self.unary_node,
            self.binary_node,
            self.binary_node.children[0],
            self.node
        ]
        leafiness = [n.is_leaf for n in nodes]
        self.assertEqual(leafiness, [True, True, True, False, False, True, False])

    def test_node_attributes(self):
        """Check node attributes"""
        correct_attributes = {
            # Current position of node
            'num'   : 0,
            'level' : 0,
            # Summary of children
            'height'   : 3,
            'node_num' : 6,
            'leaf_num' : 3,
            'edge_num' : 5,
            # Sum of subtrees
            'complexity' : 15
        }
        attributes = {
            # Current position of node
            'num'   : self.node.num,
            'level' : self.node.level,
            # Summary of children
            'height'   : self.node.height,
            'node_num' : self.node.node_num,
            'leaf_num' : self.node.leaf_num,
            'edge_num' : self.node.edge_num,
            # Sum of subtrees
            'complexity' : self.node.complexity
        }

        self.assertEqual(attributes, correct_attributes)


# Test cases
class RandomTreeTests(unittest.TestCase):
    """
    Test creation of Random Trees.
    """

    def setUp(self):
        """Setup Tests"""
        pass

    def test_random_trees(self):
        """Generate 100 random trees"""
        NUM_TREES = 500
        random_trees = [tree.random_tree(10) for i in xrange(NUM_TREES)]

        self.assertEqual(len(random_trees), NUM_TREES)

class TreeTests(unittest.TestCase):
    """
    Test the Tree Module.
    """

    def setUp(self):
        """Setup Tests"""
        self.N = 4
        self.M = 1
        self.x = np.linspace(0, float(self.M) * np.pi, self.N)
        self.y = np.linspace(0, 2.0 * float(self.M) * np.pi, self.N)
        self.TRUTH = (self.x ** 2) + (self.y ** 2)

        self.tree = tree.Tree(
            node.Node(n_add,
                node.Node('x'),
                node.Node(n_mul,
                    node.Node(10),
                    node.Node(n_abs,
                        node.Node(-4)
                    )
                )
            )
        )

    def test_tree_string(self):
        """Check tree string"""
        correct_string = '[0:0] <built-in function add> (3) - {6|15}\n    |-[1:1] x (0) - {1|1}\n    \\-[1:2] <built-in function mul> (2) - {4|8}\n          |-[2:3] 10 (0) - {1|1}\n          \\-[2:4] <built-in function abs> (1) - {2|3}\n                \\-[3:5] -4 (0) - {1|1}'
        tree_string = self.tree.display()

        self.assertEqual(tree_string, correct_string)

    def test_node_list(self):
        """Check tree node list"""
        correct_nodes = "[<built-in function add>, 'x', <built-in function mul>, 10, <built-in function abs>, -4]"
        tree_nodes = str(self.tree.list_nodes())

        self.assertEqual(str(tree_nodes), correct_nodes)

    def test_edge_list(self):
        """Check tree edge list"""
        correct_edges = "[(<built-in function add>, 'x'), (<built-in function add>, <built-in function mul>), (<built-in function mul>, 10), (<built-in function mul>, <built-in function abs>), (<built-in function abs>, -4)]"
        tree_edges = str(self.tree.list_edges())

        self.assertEqual(str(tree_edges), correct_edges)

    def test_tree_attributes(self):
        """Check tree attributes"""
        correct_attributes = {
            # Summary of children
            'height'   : 4,
            'node_num' : 6,
            'leaf_num' : 3,
            'edge_num' : 5,
            # Sum of subtrees
            'complexity' : 15
        }
        attributes = {
            # Summary of children
            'height'   : self.tree.height,
            'node_num' : self.tree.node_num,
            'leaf_num' : self.tree.leaf_num,
            'edge_num' : self.tree.edge_num,
            # Sum of subtrees
            'complexity' : self.tree.complexity
        }

        self.assertEqual(attributes, correct_attributes)

    def test_tree_get_node(self):
        """Check tree get node"""
        node = self.tree.get_node(3)

        self.assertEqual(node.value, 10)

    def test_tree_set_node(self):
        """Check tree set node"""
        self.tree.set_node(3, node.Node(123))
        correct_string = '[0:0] <built-in function add> (3) - {6|15}\n    |-[1:1] x (0) - {1|1}\n    \\-[1:2] <built-in function mul> (2) - {4|8}\n          |-[2:3] 123 (0) - {1|1}\n          \\-[2:4] <built-in function abs> (1) - {2|3}\n                \\-[3:5] -4 (0) - {1|1}'
        tree_string = self.tree.display()

        self.assertEqual(tree_string, correct_string)

    def test_tree_pruning(self):
        """Check tree pruning"""

        self.tree.prune()
