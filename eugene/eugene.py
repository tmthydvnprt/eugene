# pylint: disable=eval-used,bare-except
"""
Eugene

Operators:
===========
    Unary:
    -------
        o.abs(a)          - Same as abs(a).
        o.inv(a)          - Same as ~a.
        o.neg(a)          - Same as -a.
        o.pos(a)          - Same as +a.

    Binary:
    --------
        o.or_(a, b)       - Same as a | b.
        o.add(a, b)       - Same as a + b.
        o.and_(a, b)      - Same as a & b.
        o.div(a, b)       - Same as a / b when __future__.division is not in effect.
        o.eq(a, b)        - Same as a==b.
        o.floordiv(a, b)  - Same as a // b.
        o.ge(a, b)        - Same as a>=b.
        o.gt(a, b)        - Same as a>b.
        o.le(a, b)        - Same as a<=b.
        o.lt(a, b)        - Same as a<b.
        o.mod(a, b)       - Same as a % b.
        o.mul(a, b)       - Same as a * b.
        o.ne(a, b)        - Same as a!=b.
        o.pow(a, b)       - Same as a ** b.
        o.sub(a, b)       - Same as a - b.
        o.truediv(a, b)   - Same as a / b when __future__.division is in effect.
        o.xor(a, b)       - Same as a ^ b.

Functions:
===========
    Unary:
    -------
        np.acos(x)         - Return the arc cosine (measured in radians) of x.
        np.acosh(x)        - Return the hyperbolic arc cosine (measured in radians) of x.
        np.asin(x)         - Return the arc sine (measured in radians) of x.
        np.asinh(x)        - Return the hyperbolic arc sine (measured in radians) of x.
        np.atan(x)         - Return the arc tangent (measured in radians) of x.
        np.atanh(x)        - Return the hyperbolic arc tangent (measured in radians) of x.
        np.ceil(x)         - Return the ceiling of x as a float. This is the smallest integral value >= x.
        np.cos(x)          - Return the cosine of x (measured in radians).
        np.cosh(x)         - Return the hyperbolic cosine of x.
        np.degrees(x)      - Convert angle x from radians to degrees.
        sp.erf(x)          - Error function at x.
        sp.erfc(x)         - Complementary error function at x.
        np.exp(x)          - Return e raised to the power of x.
        np.expm1(x)        - Return exp(x)-1. This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.
        np.fabs(x)         - Return the absolute value of the float x.
        sm.factorial(x)    - Find x!. Raise a ValueError if x is negative or non-integral. -> Integral
        np.floor(x)        - Return the floor of x as a float. This is the largest integral value <= x.
        sp.gamma(x)        - Gamma function at x.
        np.isinf(x)        - Check if float x is infinite (positive or negative). -> bool
        np.isnan(x)        - Check if float x is not a number (NaN). -> bool
        sp.lgamma(x)       - Natural logarithm of absolute value of Gamma function at x.
        np.log10(x)        - Return the base 10 logarithm of x.
        np.log2(x)         - Return the base 2 logarithm of x.
        np.log1p(x)        - Return the natural logarithm of 1+x (base e). The result is computed in a way which is accurate for x near zero.
        np.log(x)          - Return the natural logarithm of x.
        np.radians(x)      - Convert angle x from degrees to radians.
        np.sin(x)          - Return the sine of x (measured in radians).
        np.sinh(x)         - Return the hyperbolic sine of x.
        np.sqrt(x)         - Return the square root of x.
        np.tan(x)          - Return the tangent of x (measured in radians).
        np.tanh(x)         - Return the hyperbolic tangent of x.
        np.trunc(x:Real)   - Truncates x to the nearest Integral toward 0. Uses the __trunc__ magic method. -> Integral

    Binary:
    --------
        np.atan2(y, x)     - Return the arc tangent (measured in radians) of y/x. Unlike atan(y/x), the signs of both x and y are considered.
        np.copysign(x, y)  - Return x with the sign of y.
        np.fmod(x, y)      - Return fmod(x, y), according to platform C.  x % y may differ.
        np.hypot(x, y)     - Return the Euclidean distance, sqrt(x*x + y*y).
        np.ldexp(x, i)     - Return x * (2**i).
        np.pow(x, y)       - Return x**y (x to the power of y).
        np.round(x[, y])   - Return the floating point value x rounded to y digits after the decimal point.

    N-ary:
    --------
        np.max(x, y, ...)  - Return the largest item in an iterable or the largest of two or more arguments.
        np.min(x, y, ...)  - Return the smallest item in an iterable or the smallest of two or more arguments.
        np.fsum([x,y,...]) - Return an accurate floating point sum of values in the iterable.
        np.prod([x,y,...]) - Return an accurate floating point product of values in the iterable.

Constants:
===========
    np.p - The mathematical constant pi = 3.141592..., to available precision.
    np.e - The mathematical constant e  = 2.718281..., to available precision.

EPHEMERAL VARIABLES : - once created stay constant, only can be used during initialization or mutation
=====================
    r.random()                 - Returns x in the interval [0, 1).
    r.randint(a, b)            - Returns integer x in range [a, b].
    r.uniform(a, b)            - Returns number x in the range [a, b) or [a, b] depending on rounding.
    r.normalvariate(mu, sigma) - Returns number x from Normal distribution. mu is the mean, and sigma is the standard deviation.

VARIABLES:
===========
    a, b, c, ..., x, y, z    - whatever you need

To add:
========
pd.shift()

removed:
=========
o.not_() - can't operate on array, but o.inv() can & produces the same result

"""

# dependancies
import sys
import time
import bisect
import tabulate
import operator as o
import random as r
import numpy as np
import pandas as pd
import copy as cp
import scipy.special as sp
import scipy.misc as sm
from multiprocessing import Pool

DEFAULT_OBJECTIVE = lambda x: 1.0
DISPLAY_NODE_STR = '%s%s[%s:%s] %s (%s) - {%s|%s}'
VARIABLES = ['x']
UNARIES = [
    'n_abs', 'n_inv', 'n_neg', 'n_pos', 'n_acos', 'n_acosh', 'n_asin', 'n_asinh', 'n_atan', 'n_atanh', 'n_ceil', 'n_cos', \
    'n_cosh', 'n_degrees', 'n_exp', 'n_expm1', 'n_fabs', 'n_factorial', 'n_floor', 'n_gamma', 'n_isinf', \
    'n_isnan', 'n_gammaln', 'n_log10', 'n_log2', 'n_log1p', 'n_log', 'n_radians', 'n_sin', 'n_sinh', 'n_sqrt', 'n_tan', \
    'n_tanh', 'n_trunc'
]
# 'n_erf', 'n_erfc',
BINARIES = [
    'n_or', 'n_add', 'n_and', 'n_div', 'n_eq', 'n_floordiv', 'n_ge', 'n_gt', 'n_le', 'n_lt', 'n_mod', 'n_mul', \
    'n_ne', 'n_sub', 'n_xor', 'n_atan2', 'n_copysign', 'n_fmod', 'n_hypot', 'n_ldexp', 'n_pow', 'n_round'
]
NARIES = ['n_max', 'n_min', 'n_sum', 'n_prod']
CONSTS = ['np.pi', 'np.e']

EPHEMERAL = {
    0: r.randint(-500, 500),
    1: r.random(),
    2: r.uniform(-500, 500),
    3: r.normalvariate(0, 100)
}

x = np.array(range(100))

# pylint: disable=invalid-name
n_abs = o.abs
n_neg = o.neg
n_pos = o.pos
# n_or = o.or_
# n_add = o.add
# n_and = o.and_
n_eq = o.eq
n_ge = o.ge
n_gt = o.gt
n_le = o.le
n_lt = o.lt
n_mul = o.mul
n_ne = o.ne
n_sub = o.sub
# n_xor = o.xor
n_acos = np.arccos
n_acosh = np.arccosh
n_asin = np.arcsin
n_asinh = np.arcsinh
n_atan = np.arctan
n_atanh = np.arctanh
n_ceil = np.ceil
n_cos = np.cos
n_cosh = np.cosh
n_degrees = np.degrees
# n_erf = sp.erf
# n_erfc = sp.erfc
n_exp = np.exp
n_expm1 = np.expm1
n_fabs = np.fabs
n_factorial = sm.factorial
n_floor = np.floor
n_gamma = sp.gamma
n_isinf = np.isinf
n_isnan = np.isnan
n_gammaln = sp.gammaln
n_log10 = np.log10
n_log2 = np.log2
n_log1p = np.log1p
n_radians = np.radians
n_sin = np.sin
n_sinh = np.sinh
n_sqrt = np.sqrt
n_tan = np.tan
n_tanh = np.tanh
n_trunc = np.trunc
n_atan2 = np.arctan2
n_copysign = np.copysign
n_fmod = np.fmod
n_hypot = np.hypot
# n_ldexp = np.ldexp
n_log = np.log
n_pow = np.power
# n_round = np.around
# pylint: enable=invalid-name

# safe functions, graceful error fallbacks
def intify(_):
    """safe intify"""
    return 1 if np.isnan(_) or not np.isfinite(_) else int(_) if not isinstance(_, 'pd.core.series.Series') else _.fillna(0).astype(int)

def n_inv(a):
    """safe inv"""
    return o.inv(intify(a))

def n_and(a, b):
    """safe and"""
    return o.and_(intify(a), intify(b))

def n_or(a, b):
    """safe or"""
    return o.or_(intify(a), intify(b))

def n_xor(a, b):
    """safe xor"""
    return o.xor(intify(a), intify(b))

def n_mod(a, b):
    """safe mod"""
    return np.where(b != 0, o.mod(a, b), 1)

def n_div(a, b):
    """safe div"""
    return np.where(b != 0, o.div(a, b), 1)

def n_floordiv(a, b):
    """safe floordiv"""
    return np.where(b != 0, o.floordiv(a, b), 1)

def n_round(a, b):
    """safe round"""
    element_round = np.vectorize(np.round)
    return element_round(a, intify(b))

def n_ldexp(a, b):
    """safe ldexp"""
    return np.ldexp(a, intify(b))

# n-ary custom functions
def n_max(_):
    """max reduction"""
    return reduce(np.maximum, _)
def n_min(_):
    """min reduction"""
    return reduce(np.minimum, _)
def n_sum(_):
    """sum reduction"""
    return reduce(np.add, _)
def n_prod(_):
    """product reduction"""
    return reduce(np.multiply, _)

class ProgressBar(object):
    """implements a comand-line progress bar"""

    def __init__(self, iterations):
        """create a progress bar"""
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)

    def animate(self, iterate):
        """animate progress"""
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iterate + 1)
        return self

    def update_iteration(self, elapsed_iter):
        """increment progress"""
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar = '%s  %s of %s complete' % (self.prog_bar, elapsed_iter, self.iterations)
        return self

    def __update_amount(self, new_amount):
        """update amout of progress"""
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[%s%s]' % ((self.fill_char * num_hashes), ' ' * (all_full - num_hashes))
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%s%%' % (percent_done)
        self.prog_bar = '%s%s%s' % (self.prog_bar[0:pct_place], pct_string, self.prog_bar[pct_place + len(pct_string):])
        return self

    def __str__(self):
        """string representation"""
        return str(self.prog_bar)

def par_fit():
    """create function to run fitness in parallel"""
    pass

def rmse(predicted, truth):
    """return the mean square error"""
    if np.isnan(predicted).any() or predicted.shape != truth.shape:
        result = np.inf
    else:
        result = np.sqrt(((predicted - truth) ** 2).mean())
    return result

def random_tree(max_level=20, min_level=1, current_level=0):
    """generate a random tree"""
    return Tree(random_node(max_level, min_level, current_level))

def random_node(max_level=20, min_level=1, current_level=0):
    """node = a random node or nodes"""
    # pylint: disable=too-many-branches
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
        rand_node = r.randint(4, 6) if current_level < min_level else r.randint(0, 6)
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
        # node = a n-ary operator
        elif rand_node == 6:
            nary_node_num = r.randint(2, 5)
            if nary_node_num == 2:
                node = Node(
                    NARIES[r.randint(0, len(NARIES) - 1)],
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1)
                )
            elif nary_node_num == 3:
                node = Node(
                    NARIES[r.randint(0, len(NARIES) - 1)],
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1)
                )
            elif nary_node_num == 4:
                node = Node(
                    NARIES[r.randint(0, len(NARIES) - 1)],
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1)
                )
            elif nary_node_num == 5:
                node = Node(
                    NARIES[r.randint(0, len(NARIES) - 1)],
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1),
                    random_node(max_level - 1, current_level + 1)
                )
    return node
    # pylint: enable=too-many-branches

class Node(object):
    """
    Defines a node of the tree
    """

    def __init__(self, value=None, *children):
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
        """check if this node is a leaf"""
        return len(self.children) == 0

    @property
    def ary(self):
        """return the arity of the node"""
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

    def set_nums(self, node_counter=-1, level_counter=0, leaf_count=-1, edge_count=-1):
        """set node numbers (depth first)"""

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

class Tree(object):
    """
    Defined a Tree with nodes
    """

    def __init__(self, nodes=None, subtree=False):
        self.nodes = nodes
        if not subtree:
            self.nodes.set_nums()

    @property
    def height(self):
        """return the number of levels in the tree"""
        return self.nodes.height + 1

    @property
    def node_num(self):
        """return the number of nodes in the tree"""
        return self.nodes.node_num

    @property
    def leaf_num(self):
        """return the number of leaves in the tree"""
        return self.nodes.leaf_num + 1

    @property
    def edge_num(self):
        """return the number of edges in the tree"""
        return self.nodes.edge_num + 1

    @property
    def complexity(self):
        """return the complexity of the tree (sum of nodes in tree and each subtree)"""
        return self.nodes.complexity

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.nodes)

    def evaluate(self):
        """evaluate expression stored in tree"""
        try:
            result = np.array(eval(compile(self.__str__(), '', 'eval')))
        except:
            result = np.nan
        return result

    def get_node(self, n=0):
        """return a node from the tree"""

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

    def set_node(self, n=0, node=None):
        """set a node in the tree"""

        # search tree until node number is found, and store sub tree
        if self.nodes.num == n:
            self.nodes = node
        else:
            self.nodes.children = tuple([Tree(c, subtree=True).set_node(n, node) for c in self.nodes.children])

        # rebase the numbers of the Tree
        self.nodes.set_nums()
        return self.nodes

    def list_edges(self):
        """get edges of tree"""

        # get list of tuple edges between nodes e.g. [(n1,n2),(n1,n3)...]
        edges = [(self.nodes.value, c.value if len(c.children) > 0 else c.value) for c in self.nodes.children]
        children_nodes = [Tree(c, subtree=True).list_edges() for c in self.nodes.children if len(c.children) > 0]
        for i in xrange(len(children_nodes)):
            edges += children_nodes[i]
        return edges

    def list_nodes(self):
        """return nodes of tree"""

        # get list of nodes
        node_list = []
        node_list.append(self.nodes.value)
        # add children
        node_list.extend([c.value for c in self.nodes.children if len(c.children) == 0])
        # add children's children
        grand_children = [Tree(c, subtree=True).list_nodes() for c in self.nodes.children if len(c.children) > 0]
        node_list.extend([node for grand_child in grand_children for node in grand_child])

        return node_list

    def prune(self):
        """go thru nodes and remove or replace dead / constant branches (subtrees)"""

        sub_tree = Tree(self.nodes, subtree=True)
        contains_variable = any([n in VARIABLES for n in sub_tree.list_nodes()])

        if not contains_variable:
            sub_eval = sub_tree.evaluate()
            # node properties
            self.nodes.value = sub_eval
            self.nodes.children = ()

        else:
            for child in self.nodes.children:
                Tree(child, subtree=True).prune()

        # rebase the numbers of the Tree
        self.nodes.set_nums()
        return self.nodes

    def display(self, level=0, level_list=None):
        """display helper"""
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

class Individual(object):
    """docstring for Individual"""

    def __init__(self, chromosomes=None):
        self.chromosomes = chromosomes

    @property
    def size(self):
        """return size of individual"""
        return self.chromosomes.node_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.chromosomes)

    def display(self):
        """display helper"""
        self.chromosomes.display()

    def compute_gene_expression(self, error_function=None, target=None):
        """compute gene expression by evaluating function stored in tree, and keep track of time"""

        # evaluate function and time to compute
        t0 = time.time()
        output = self.chromosomes.evaluate()
        t1 = time.time()

        # calculate error of result and time complexity
        error = error_function(output, target)
        time_complexity = t1 - t0
        physical_complexity = self.chromosomes.complexity

        return np.array([error, time_complexity, physical_complexity])

    def crossover(self, spouse=None):
        """randomly crossover two chromosomes"""

        # create random crossover points
        x1 = r.randint(0, self.size - 1)
        x2 = r.randint(0, spouse.size - 1)

        # clone parent chromosomes
        c1 = cp.deepcopy(self.chromosomes)
        c2 = cp.deepcopy(spouse.chromosomes)

        # get nodes to cross
        c1n = c1.get_node(x1)
        c2n = c2.get_node(x2)

        # transfer nodes
        c1.set_node(x1, c2n)
        c2.set_node(x2, c1n)

        return (Individual(c1), Individual(c2))

    def mutate(self):
        """ alter a random node in chromosomes"""

        # randomly select node to mutate
        mpoint = r.randint(0, self.size - 1)

        # mutate whole node by replacing children with random subtree
        if r.random() >= 0.5:
            rand_tree = random_tree(2)
            x2 = r.randint(0, rand_tree.node_num - 1)
            node = rand_tree.get_node(x2)
            self.chromosomes.set_node(mpoint, node)
            # check and prune tree with new subtree for inefficiencies
            self.chromosomes.prune()

        # or just mutate node value based on current type
        else:
            node = self.chromosomes.get_node(mpoint)
            # constant
            if node.value in CONSTS:
                mutated_value = CONSTS[r.randint(0, len(CONSTS) - 1)]
            # variable
            elif node.value in VARIABLES:
                mutated_value = VARIABLES[r.randint(0, len(VARIABLES) - 1)]
            # a unary operator
            elif node.value in UNARIES:
                mutated_value = UNARIES[r.randint(0, len(UNARIES) - 1)]
            # a binary operator
            elif node.value in BINARIES:
                mutated_value = BINARIES[r.randint(0, len(BINARIES) - 1)]
            # a n-ary operator
            elif node.value in NARIES:
                mutated_value = NARIES[r.randint(0, len(NARIES) - 1)]
            # EPHEMERAL constant random ( 0:1, uniform -500:500, or normal -500:500 )
            else:
                mutated_value = EPHEMERAL[r.randint(1, len(EPHEMERAL) - 1)]

            # mutate node value (keeps children, if applicable)
            node.value = mutated_value
            self.chromosomes.set_node(mpoint, node)


class Population(object):
    """Defines Population of Individuals with ability to create generations and evaluate fitness"""
    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-public-methods
    def __init__(
            self,
            init_population_size=1000,
            objective_function=None,
            error_function=rmse,
            target=None,
            max_generations=1000,
            init_tree_size=3,
            stagnation_timeout=20,
            rank_pressure=2.0,
            elitism=0.02,
            replication=0.28,
            mating=0.6,
            mutation=0.1,
            parallel=False
        ):
        # parameters
        self.init_population_size = init_population_size
        self.objective_function = objective_function
        self.error_function = error_function
        self.target = target
        self.max_generations = max_generations
        self.init_tree_size = init_tree_size
        self.stagnation_timeout = stagnation_timeout
        self.rank_pressure = rank_pressure
        self.elitism = elitism
        self.replication = replication
        self.mating = mating
        self.mutation = mutation
        self.parallel = parallel
        # initialize variables
        self.created = False
        self.individuals = []
        self.ranking = []
        self.generation = 0
        self.expression_scale = np.array([1.0, 1.0, 1.0])
        self.history = {
            'fitness' : [],
            'error' : [],
            'time' : [],
            'complexity' : []
        }
        # cached values
        self._fitness = np.array([])
    # pylint: enable=too-many-arguments,too-many-instance-attributes,too-many-public-methods

    @property
    def size(self):
        """return the size of the population"""
        return len(self.individuals)

    @property
    def fitness(self):
        """return the fitness of each individual in population"""
        if self._fitness.shape == (0, ):
            self.calc_fitness()
        return self._fitness

    @property
    def stagnate(self):
        """
        determine if the population has stagnated and reached local min
        where average fitness over last n generations has not changed
        """
        if self.generation <= self.stagnation_timeout:
            return False
        else:
            last_gen2 = self.history['fitness'][(self.generation - 2 - self.stagnation_timeout):(self.generation - 2)]
            last_gen1 = self.history['fitness'][(self.generation - 1 - self.stagnation_timeout):(self.generation - 1)]
            return (last_gen2 == last_gen1) and not np.isinf(last_gen1).all()

    def describe(self):
        """print out all data"""
        self.describe_init()
        print '\n'
        self.describe_current()

    def describe_init(self):
        """print out parameters used to intialize population"""
        print '\nPopulation Initialized w/ Parameters:'
        data = [
            ['Initial number of individuals:', self.init_population_size],
            ['Initial size of individuals:', self.init_tree_size],
            ['Max. number of generations:', self.max_generations],
            ['Parallel fitness turned on:', self.parallel],
            ['Stagnation factor:', self.stagnation_timeout],
            ['Percent of Elitism:', self.elitism],
            ['Percent of replication:', self.replication],
            ['Percent of mating:', self.mating],
            ['Percent of mutation:', self.mutation],
            ['Selective pressure:', self.rank_pressure]
        ]
        print tabulate.tabulate(data)

    def describe_current(self):
        """print out status about current population"""
        print '\nCurrent Population Status:'
        # initialize VARIABLES
        data = [
            ['Current generation:', self.generation],
            ['Number of individuals:', self.size]
        ]
        # if fitness is empty
        if self.fitness.shape == (0,):
            data.extend([
                ['Max fitness:', 0.0],
                ['Average fitness:', 0.0],
                ['Min fitness:', 0.0],
            ])
        else:
            data.extend([
                ['Max fitness:', self.fitness.max()],
                ['Average fitness:', self.fitness.mean()],
                ['Min fitness:', self.fitness.min()],
            ])
        print tabulate.tabulate(data)
        if self.fitness.max() == self.fitness.mean() == self.fitness.min():
            print '\n'
            print 'Constant Fitness: It looks like the template objective functions is being used!'
            print '                  Please add your own with '

    def initialize(self, seed=None):
        """initialize a population based on seed or randomly"""

        self.describe_init()
        self.created = True
        if seed:
            print '\nUsing seed for inital population'
            self.individuals = seed
        else:
            print '\nInitializing Population with Individuals composed of random Trees:'
            pb = ProgressBar(self.init_population_size)
            while len(self.individuals) < self.init_population_size:
                # generate a random expression tree
                tree = random_tree(self.init_tree_size)
                # prune inefficiencies from the tree
                tree.prune()
                # create an individual from this expression
                individual = Individual(tree)
                # check for genes
                gene_expression = individual.compute_gene_expression(self.error_function, self.target)
                # if there is some non-infinite error, add to the population
                if not np.isinf(gene_expression[0]):
                    self.individuals.append(individual)
                    pb.animate(self.size)
        print '\n'
        self.describe_current()

    def calc_fitness(self):
        """calculate the fitness of each individual."""

        if self.parallel:
            pool = Pool()
            fitness = pool.map(par_fit, [(i, self.objective_function) for i in self.individuals])
            pool.close()
            pool.join()
            self._fitness = np.array(fitness)

        else:
            expression = np.array([i.compute_gene_expression(self.error_function, self.target) for i in self.individuals])
            expression_scale = np.array(np.ma.masked_invalid(expression).max(axis=0))
            max_expr = np.array(np.ma.masked_invalid(expression).max(axis=0)) / expression_scale
            mean_expr = np.array(np.ma.masked_invalid(expression).mean(axis=0)) / expression_scale
            min_expr = np.array(np.ma.masked_invalid(expression).min(axis=0)) / expression_scale

            self._fitness = self.objective_function(expression, expression_scale)
            mfit = np.ma.masked_invalid(self._fitness)

            self.history['fitness'].append((mfit.max(), mfit.mean(), mfit.min()))
            self.history['error'].append((max_expr[0], mean_expr[0], min_expr[0]))
            self.history['time'].append((max_expr[1], mean_expr[1], min_expr[1]))
            self.history['complexity'].append((max_expr[2], mean_expr[2], min_expr[2]))

    def rank(self):
        """create ranking of individuals"""
        self.ranking = zip(self.fitness, self.individuals)
        self.ranking.sort()

    def roulette(self, number=None):
        """select parent pairs based on roulette method (probability proportional to fitness)"""
        number = number if number else self.size
        selections = []

        # unpack
        ranked_fitness, ranked_individuals = (list(i) for i in zip(*self.ranking))
        ranked_fitness = np.array(ranked_fitness)

        # calculate weighted probability proportial to fitness
        fitness_probability = ranked_fitness / np.ma.masked_invalid(ranked_fitness).sum()
        cum_prob_dist = np.array(np.ma.masked_invalid(fitness_probability).cumsum())

        # randomly select two individuals with weighted probability proportial to fitness
        selections = [ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])] for _ in xrange(number)]
        return selections

    def stochastic(self, number=None):
        """select parent pairs based on stochastic method (probability uniform across fitness)"""
        number = number if number else self.size

        # unpack
        ranked_fitness, ranked_individuals = (list(i) for i in zip(*self.ranking))
        ranked_fitness = np.array(ranked_fitness)

        # calculate weighted probability proportial to fitness
        fitness_probability = ranked_fitness / np.ma.masked_invalid(ranked_fitness).sum()
        cum_prob_dist = np.array(np.ma.masked_invalid(fitness_probability).cumsum())

        # determine uniform points
        p_dist = 1 / float(number)
        p0 = p_dist * r.random()
        points = p0 + p_dist * np.array(range(0, number))

        # randomly select individuals with weighted probability proportial to fitness
        selections = [ranked_individuals[bisect.bisect(cum_prob_dist, p * cum_prob_dist[-1])] for p in points]
        return selections

    def tournament(self, number=None, tournaments=4):
        """select parent pairs based on tournament method (random tournaments amoung individuals where fitness wins)"""
        number = number if number else self.size
        selections = []
        for _ in xrange(number):
            # select group of random competitors
            competitors = [self.ranking[i] for i in list(np.random.random_integers(0, self.size-1, tournaments))]
            # group compete in fitness tournament (local group sorting)
            competitors.sort()
            # select most fit from each group
            winner = competitors[-1]
            selections.append(winner[1])
        return selections

    def rank_roulette(self, number=None, pressure=2):
        """select parent pairs based on rank roulette method (probability proportional to fitness rank)"""
        number = number if number else self.size
        selections = []

        # unpack
        _, ranked_individuals = (list(i) for i in zip(*self.ranking))

        # create a scaled rank by fitness (individuals already sorted, so just create rank range, then scale)
        n = self.size
        rank = range(1, n + 1)
        scaled_rank = 2.0 - pressure + (2.0 * (pressure - 1) * (np.array(rank) - 1) / (n - 1))

        # calculate weighted probability proportial to scaled rank
        scaled_rank_probability = scaled_rank / np.ma.masked_invalid(scaled_rank).sum()
        cum_prob_dist = np.array(np.ma.masked_invalid(scaled_rank_probability).cumsum())

        for _ in xrange(number):
            # randomly select individuals with weighted probability proportial to scaled rank
            p1 = ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])]
            selections.append(p1)
        return selections

    def select(self, number=None):
        """select individuals thru various methods"""
        selections = self.roulette(number)
        return selections

    def create_generation(self):
        """create the next generations, this is main function that loops"""

        # determine fitness of current generations and log average fitness
        self.calc_fitness()

        # rank individuals by fitness
        self.rank()

        # create next generation
        elite_num = int(round(self.size * self.elitism))
        replicate_num = int(round(self.size * self.replication))
        mate_num = int(round(self.size * self.mating))
        mutate_num = int(round(self.size * self.mutation))
        # split mate_num in half_size (2 parents = 2 children)
        mate_num = int(round(mate_num/2.0))

        # propogate elite
        next_generation = [i[1] for i in self.ranking[-elite_num:]]

        # replicate
        next_generation.extend(self.select(replicate_num))

        # crossover mate
        parent_pairs = zip(self.select(mate_num), self.select(mate_num))
        child_pairs = [p1.crossover(p2) for p1, p2 in parent_pairs]
        children = [child for pair in child_pairs for child in pair]
        next_generation.extend(children)

        # mutate
        mutants = self.select(mutate_num)
        for m in mutants:
            m.mutate()
        next_generation.extend(mutants)

        self.individuals = next_generation[:self.size]

        # clear cached values
        self._fitness = np.array([])

        # log generation
        self.generation += 1

        return None

    def run(self, number_of_generations=None):
        """run algorithm"""

        number_of_generations = number_of_generations if number_of_generations else self.max_generations
        pb = ProgressBar(number_of_generations)

        while self.generation < number_of_generations and not self.stagnate:
            self.create_generation()
            pb.animate(self.generation)

        if self.stagnate:
            print 'population became stagnate'
        print self.generation, 'generations'

    def most_fit(self):
        """return the most fit individual"""

        # make sure the individuals have been ranked
        self.rank()
        return self.ranking[-1][1]
