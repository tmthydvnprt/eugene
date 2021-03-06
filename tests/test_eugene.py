"""
Eguene.py - tests the eugene package.
"""

import unittest

import copy
import numpy as np

import eugene.Config

from eugene.Util import rmse, ProgressBar
from eugene.Primatives import *

from eugene.Node import Node, random_node
from eugene.Tree import Tree, random_tree
from eugene.Individual import Individual
from eugene.Population import Population

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

    def test_01_rmse(self):
        """Test that RMSE is calculated"""

        error = rmse(self.predicted, self.truth)
        self.assertEqual(self.error, error)

    def test_02_rmse_shape(self):
        """Test that RMSE fails when shapes do not match"""

        error = rmse(self.predicted[:-2], self.truth)
        self.assertEqual(np.inf, error)

    def test_03_rmse_nan(self):
        """Test that RMSE fails when NaN is present"""

        predicted = self.predicted
        predicted[4] = np.NaN
        error = rmse(predicted, self.truth)
        self.assertEqual(np.inf, error)

    def test_04_rmse_datatype(self):
        """Test that RMSE fails when wrong data type is passed"""

        error = rmse(list(self.predicted), self.truth)
        self.assertEqual(np.inf, error)

    def test_05_progress_bar(self):
        """Run the progress bar"""

        p = ProgressBar(len(self.predicted))

        for i in xrange(len(self.predicted)):
            p.animate(i)

class RandomNodeTests(unittest.TestCase):
    """
    Test creation of Random Nodes.
    """

    def setUp(self):
        """Setup Tests"""
        pass

    def test_06_random_trees(self):
        """Generate 500 random nodes"""
        NUM_NODES = 500
        random_nodes = [random_node(10) for i in xrange(NUM_NODES)]

        self.assertEqual(len(random_nodes), NUM_NODES)

class NodeTests(unittest.TestCase):
    """
    Test Node Module.
    """

    def setUp(self):
        """Setup Tests"""
        self.variable_node = Node('x')
        self.constant_node = Node(3)
        self.ephemeral_node = Node(EPHEMERAL[0])
        self.unary_node = Node('n_abs', Node(-3))
        self.binary_node = Node('n_add', Node(-3), Node(10))
        self.node = Node('n_add',
            Node('x'),
            Node('n_mul',
                Node(10),
                Node('n_abs',
                    Node(-4)
                )
            )
        )

        self.variable_node.set_nums()
        self.constant_node.set_nums()
        self.ephemeral_node.set_nums()
        self.unary_node.set_nums()
        self.binary_node.set_nums()
        self.node.set_nums()

    def test_07_node_numbers(self):
        """Check node numbers"""
        node_nums = self.node.set_nums()
        self.assertEqual(node_nums, (5, 6, 4, 3, 5, 15))

    def test_08_node_string(self):
        """Check node string printing"""
        correct_str = 'n_add(eugene.Config.VAR[\'x\'], n_mul(10, n_abs(-4)))'
        node_str = self.node.__repr__()
        self.assertEqual(node_str, correct_str)

    def test_09_node_arity(self):
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

    def test_10_node_leafiness(self):
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

    def test_11_node_attributes(self):
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

    def test_12_random_trees(self):
        """Generate 500 random trees"""
        NUM_TREES = 500
        random_trees = [random_tree(10) for i in xrange(NUM_TREES)]

        self.assertEqual(len(random_trees), NUM_TREES)

class TreeTests(unittest.TestCase):
    """
    Test the Tree Module.
    """

    def setUp(self):
        """Setup Tests"""

        eugene.Config.VAR = {'x' : np.arange(0, 5)}
        eugene.Config.truth = eugene.Config.VAR['x'] + (10 * abs(-4))

        self.tree = Tree(
            Node('n_add',
                Node('x'),
                Node('n_mul',
                    Node(10),
                    Node('n_abs',
                        Node(-4)
                    )
                )
            )
        )

    def test_13_tree_string(self):
        """Check tree string"""
        correct_string = '[0:0] n_add (3) - {6|15}\n    |-[1:1] x (0) - {1|1}\n    \\-[1:2] n_mul (2) - {4|8}\n          |-[2:3] 10 (0) - {1|1}\n          \\-[2:4] n_abs (1) - {2|3}\n                \\-[3:5] -4 (0) - {1|1}'
        tree_string = self.tree.display()
        self.assertEqual(repr(self.tree), str(self.tree))
        self.assertEqual(tree_string, correct_string)

    def test_14_node_list(self):
        """Check tree node list"""
        correct_nodes = "['n_add', 'x', 'n_mul', 10, 'n_abs', -4]"
        tree_nodes = str(self.tree.list_nodes())

        self.assertEqual(str(tree_nodes), correct_nodes)

    def test_15_edge_list(self):
        """Check tree edge list"""
        correct_edges = "[('n_add', 'x'), ('n_add', 'n_mul'), ('n_mul', 10), ('n_mul', 'n_abs'), ('n_abs', -4)]"
        tree_edges = str(self.tree.list_edges())

        self.assertEqual(str(tree_edges), correct_edges)

    def test_16_tree_attributes(self):
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

    def test_17_evaluate(self):
        """Evaluate Tree expression"""

        result = self.tree.evaluate()

        self.assertEqual(result.tolist(), eugene.Config.truth.tolist())

    def test_18_evaluate(self):
        """Evaluate Tree expression with nan in variable"""

        eugene.Config.VAR['x'] = np.NaN
        result = self.tree.evaluate()

        self.assertTrue(np.isnan(result))

    def test_19_evaluate(self):
        """Evaluate Tree expression with no set variable"""

        eugene.Config.VAR = {}
        result = self.tree.evaluate()

        self.assertTrue(np.isnan(result))

    def test_20_tree_get_node(self):
        """Check tree get node"""
        node = self.tree.get_node(3)

        self.assertEqual(node.value, 10)

    def test_21_tree_set_node(self):
        """Check tree set node"""
        self.tree.set_node(3, Node(123))
        correct_string = '[0:0] n_add (3) - {6|15}\n    |-[1:1] x (0) - {1|1}\n    \\-[1:2] n_mul (2) - {4|8}\n          |-[2:3] 123 (0) - {1|1}\n          \\-[2:4] n_abs (1) - {2|3}\n                \\-[3:5] -4 (0) - {1|1}'
        tree_string = self.tree.display()

        self.assertEqual(tree_string, correct_string)

    def test_22_tree_pruning(self):
        """Check tree pruning"""

        self.tree.prune()

class IndividualTests(unittest.TestCase):
    """
    Test the Individual Module.
    """

    def setUp(self):
        """Setup Tests"""
        N = 4
        M = 1
        eugene.Config.VAR = {
            'x' : np.linspace(0, float(M) * np.pi, 1),
            'y' :  np.linspace(0, 2.0 * float(M) * np.pi, 4)
        }
        eugene.Config.TRUTH = (eugene.Config.VAR['x'] ** 2.0) + (eugene.Config.VAR['y'] ** 2.0)

        self.ind1 = Individual(Tree(
            Node('n_add',
                Node('n_pow',
                    Node('x'),
                    Node(2.0)
                ),
                Node('n_pow',
                    Node('y'),
                    Node(2.0)
                )
            )
        ))
        self.ind2 = Individual(Tree(
            Node('n_add',
                Node('x'),
                Node('n_mul',
                    Node(10),
                    Node('n_abs',
                        Node(-4)
                    )
                )
            )
        ))

    def test_23_individual_size(self):
        """Check Individual size"""
        self.assertEqual(self.ind1.size, 7)

    def test_24_gene_expression(self):
        """Evaluate Gene Expression"""
        error, time, complexity = self.ind1.compute_gene_expression(rmse, eugene.Config.TRUTH)
        self.assertEqual(error, 0.0)
        self.assertEqual(complexity, 17.0)

        # def error_and_complexity(gene_expression, scale):
        #     """user fitness function, weighted combination of error and complexity"""
        #     print gene_expression
        #     weights = np.array([0.95, 0.025, 0.025])
        #     scaled_gene_expression = 1.0 / (gene_expression / scale)
        #
        #     return np.dot(scaled_gene_expression, weights)

    def test_25_mutate(self):
        """Test individual mutation"""
        ind1_orig = copy.deepcopy(self.ind1)
        self.assertEqual(repr(ind1_orig), repr(self.ind1))
        for _ in xrange(50):
            self.ind1.mutate()
        self.assertNotEqual(repr(ind1_orig), repr(self.ind1))

    def test_26_mutate(self):
        """Test individual mutation with pruning"""
        ind1_orig = copy.deepcopy(self.ind1)
        self.assertEqual(repr(ind1_orig), repr(self.ind1))
        for _ in xrange(20):
            self.ind1.mutate(pruning=True)
        self.assertNotEqual(repr(ind1_orig), repr(self.ind1))

    def test_27_crossover(self):
        """Test individual crossover"""
        self.assertNotEqual(repr(self.ind1), repr(self.ind2))
        child1, child2 = self.ind1.crossover(self.ind2)
        self.assertNotEqual(repr(child1), repr(self.ind1))
        self.assertNotEqual(repr(child2), repr(self.ind2))
        self.assertNotEqual(repr(child1), repr(child2))

    def test_28_individual_string(self):
        """Check individual display string"""
        correct_string = '[0:0] n_add (2) - {7|17}\n    |-[1:1] n_pow (1) - {3|5}\n    |     |-[2:2] x (0) - {1|1}\n    |     \\-[2:3] 2.0 (0) - {1|1}\n    \\-[1:4] n_pow (1) - {3|5}\n          |-[2:5] y (0) - {1|1}\n          \\-[2:6] 2.0 (0) - {1|1}'
        tree_string = self.ind1.display()

        self.assertEqual(tree_string, correct_string)

class PopulationTests(unittest.TestCase):
    """
    Test the Population Module.
    """

    def setUp(self):
        """Setup Tests"""

        # Setup up variable and truth configuration
        eugene.Config.VAR['x'] = np.linspace(0, 8.0 * np.pi, 1024)
        eugene.Config.TRUTH = eugene.Config.VAR['x'] * np.sin(eugene.Config.VAR['x']) + eugene.Config.VAR['x']/2.0 + 1.61

        def error_and_complexity(gene_expression, scale):
            """user fitness function, weighted combination of error and complexity"""

            weights = np.array([0.95, 0.025, 0.025])
            scaled_gene_expression = 1.0 / (gene_expression / scale)

            return np.dot(scaled_gene_expression, weights)

        # Setup Populations
        self.P1 = Population(
            init_population_size=100,
            objective_function=error_and_complexity,
            max_generations=100,
            init_tree_size=2,
            target=eugene.Config.TRUTH,
            pruning=False
        )

    def test_29_initialize_population(self):
        """Initialize a Random Population"""
        self.P1.initialize()

    def test_30_initialize_population(self):
        """Initialize a seeded Population"""
        inds = [Individual(Tree(
            Node('n_add',
                Node('n_pow',
                    Node('x'),
                    Node(2.0)
                ),
                Node('n_pow',
                    Node('y'),
                    Node(2.0)
                )
            )
        ))]
        self.P1.initialize(inds)

    def test_31_display_population(self):
        """Display Population Info"""
        self.P1.initialize()
        self.P1.describe()

    def test_32_run_population_roulette(self):
        """Run Population for 10 generations with roulette selection"""
        self.P1.initialize()
        self.P1.run(10)

    def test_33_run_population_pruning(self):
        """Run Population for 10 generations with pruning"""
        self.P1.pruning = True
        self.P1.initialize()
        self.P1.run(10)

    def test_34_run_population_stochastic(self):
        """Run Population for 10 generations with stochastic selection"""
        self.P1.selectmethod = 'stochastic'
        self.P1.initialize()
        self.P1.run(10)

    def test_35_run_population_tournament(self):
        """Run Population for 10 generations with tournament selection"""
        self.P1.selectmethod = 'tournament'
        self.P1.initialize()
        self.P1.run(10)

    def test_36_run_population_rank_roulette(self):
        """Run Population for 10 generations with rank roulette selection"""
        self.P1.selectmethod = 'rank_roulette'
        self.P1.initialize()
        self.P1.run(10)
