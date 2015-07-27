# pylint: disable=eval-used
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
import bisect
import operator as o
import random as r
import numpy as np
import pandas as pd
import copy as cp
import scipy.special as sp
import scipy.misc as sm
import tabulate

VARIABLES = ['x']
x = np.zeros(100)

UNARIES = [
    'n_abs', 'n_inv', 'n_neg', 'n_pos', 'n_acos', 'n_acosh', 'n_asin', 'n_asinh', 'n_atan', 'n_atanh', 'n_ceil', 'n_cos', \
    'n_cosh', 'n_degrees', 'n_erf', 'n_erfc', 'n_exp', 'n_expm1', 'n_fabs', 'n_factorial', 'n_floor', 'n_gamma', 'n_isinf', \
    'n_isnan', 'n_gammaln', 'n_log10', 'n_log2', 'n_log1p', 'n_log', 'n_radians', 'n_sin', 'n_sinh', 'n_sqrt', 'n_tan', \
    'n_tanh', 'n_trunc'
]
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
n_erf = sp.erf
n_erfc = sp.erfc
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
    return pd.Series(np.where(b != 0, o.mod(a, b), 1))

def n_div(a, b):
    """safe div"""
    return pd.Series(np.where(b != 0, o.div(a, b), 1))

def n_floordiv(a, b):
    """safe floordiv"""
    return pd.Series(np.where(b != 0, o.floordiv(a, b), 1))

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

class Node(object):
    """
    Defines a node of the tree
    """

    def __init__(self, value=None, *children):
        self.value = value
        self.children = children
        self.num = None
        self.total = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # node is a variable or constant
        if len(self.children) == 0:
            return str(self.value)
        # node is a unary, binary or n-ary function
        else:
            if self.value in NARIES:
                return str(self.value) + '([' + ','.join([str(c) for c in self.children]) + '])'
            else:
                return str(self.value) + '(' + ','.join([str(c) for c in self.children]) + ')'

class Tree(object):
    """
    Defined a Tree with nodes
    """

    def __init__(self, nodes=None):
        self.nodes = nodes
        #self.set_nums()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.nodes)

    def evaluate(self):
        """evaluate expression stored in tree"""
        try:
            return np.array(eval(compile(self.__str__(), '', 'eval')))
        except:
            return np.zeros(x.shape)

    def set_nums(self, count=-1):
        """set node numbers (depth first)"""
        count += 1
        self.nodes.num = count
        if len(self.nodes.children) > 0:
            for c in self.nodes.children:
                count = Tree(c).set_nums(count)
        self.nodes.total = count
        return count

    def get_node(self, n=0):
        """return a node"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()
        # search tree until node number is found and take sub tree
        if self.nodes.num == n:
            return cp.deepcopy(self.nodes)
        elif len(self.nodes.children) > 0:
            for c in self.nodes.children:
                cn = Tree(c).get_node(n)
                if cn:
                    return cn
        else:
            return None

    def set_node(self, n=0, node=None):
        """set a node in the tree"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()
        # search tree until node number is found, and store sub tree
        if self.nodes.num == n:
            self.nodes = node
        else:
            self.nodes.children = tuple([Tree(c).set_node(n, node) for c in self.nodes.children])
        return self.nodes

    def display(self, level=0, level_list=None):
        """display helper"""
        level_list = level_list if level_list else []
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0 or level == 0:
            self.set_nums()
        if level == 0:
            node_str = '[0:' + str(self.nodes.num) + '] ' + str(self.nodes.value)
        else:
            if level_list[-1] == '      ':
                node_str = '    ' + ''.join(level_list[:-1]) + r'\-[' + str(level) +':'+ str(self.nodes.num) +'] '+ str(self.nodes.value)
            else:
                node_str = '    ' + ''.join(level_list[:-1]) + r'|-[' + str(level) +':'+ str(self.nodes.num) +'] '+ str(self.nodes.value)
        print node_str
        for i, child in enumerate(self.nodes.children):
            Tree(child).display(level+1, level_list + ['      ' if i == len(self.nodes.children) - 1 else '|     '])

    def list_edges(self):
        """get edges of tree"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()
        # get list of tuple edges between nodes e.g. [(n1,n2),(n1,n3)...]
        edges = [(self.nodes.value + str(self.nodes.num), c.value + str(c.num) if len(c.children) > 0 else c.value) for c in self.nodes.children]
        children_nodes = [Tree(c).list_edges() for c in self.nodes.children if len(c.children) > 0]
        for i in xrange(len(children_nodes)):
            edges += children_nodes[i]
        return edges

    def list_nodes(self):
        """return nodes of tree"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()

        # get list of nodes
        node_list = []
        node_list.append('[%s]%s' % (self.nodes.num, self.nodes.value))
        # add children
        node_list.extend(['[%s]%s' % (c.num, c.value) for c in self.nodes.children if len(c.children) == 0])
        # add children's children
        grand_children = [Tree(c).list_nodes() for c in self.nodes.children if len(c.children) > 0]
        node_list.extend([node for grand_child in grand_children for node in grand_child])

        return node_list

def random_tree(max_level=20, min_level=1, current_level=0):
    """generate a random tree"""
    return Tree(random_node(max_level, min_level, current_level))

def random_node(max_level=20, min_level=1, current_level=0):
    """node = a random node or nodes"""
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

DEFAULT_OBJECTIVE = lambda x: 1.0

class Individual(object):
    """docstring for Individual"""

    def __init__(self, chromosomes=None):
        self.chromosomes = chromosomes
        self.size = self.chromosomes.nodes.total

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.chromosomes)

    def fitness(self, objective_function=None):
        """return the fitness of an Individual based on the objective_function"""
        objective_function = objective_function if objective_function else DEFAULT_OBJECTIVE
        # compute gene expression by evaluate function stored in tree
        gene_expression = self.chromosomes.evaluate()
        # remove NaNs and Infs
        gene_expression[np.isnan(gene_expression)] = 0.0
        gene_expression[np.isinf(gene_expression)] = 0.0
        # objectively determine fitness
        return objective_function(gene_expression)

    def mate(self, spouse=None, mutate_probability=0.15):
        """mate this Individual with a spouse"""

        # perform genetic exchange
        child1, child2 = self.crossover(spouse)

        # probabilistically mutate
        if r.random() >= mutate_probability:
            child1.mutate()
        if r.random() >= mutate_probability:
            child2.mutate()

        return (child1, child2)

    def crossover(self, spouse=None):
        """randomly crossover two chromosomes"""

        # create random crossover points
        x1 = r.randint(0, self.size)
        x2 = r.randint(0, spouse.size)

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
        mpoint = r.randint(0, self.size)
        node = self.chromosomes.get_node(mpoint)

        # determine how node can mutate based on node type
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

    def __init__(self, init_population_size=1000, objective_function=None, init_tree_size=3, min_fitness=0.0, max_generations=10000, stagnation_factor=20, minimiz_fitness=False, rank_pressure=2.0):
        # parameters
        self.init_population_size = init_population_size
        self.init_tree_size = init_tree_size
        self.min_fitness = min_fitness
        self.max_generations = max_generations
        self.stagnation_factor = stagnation_factor
        self.minimiz_fitness = minimiz_fitness
        self.rank_pressure = rank_pressure
        self.objective_function = objective_function
        # initialize VARIABLES
        self.created = False
        self.individuals = []
        self.ranking = []
        self.generation = 0
        self.fitness_record = list(np.zeros(max_generations))
        # cached values
        self._fitness = np.array([])

    @property
    def size(self):
        """return the size of the population"""
        return len(self.individuals)

    @property
    def fitness(self):
        """return the fitness of each individual in population"""
        if self._fitness.shape == (0,):
            self._fitness = np.array([i.fitness(self.objective_function) for i in self.individuals])
            self._fitness[np.isnan(self._fitness)] = 0.0
            self._fitness[np.isinf(self._fitness)] = 0.0
        return self._fitness

    @property
    def stagnate(self):
        """
        determine if the population has stagnated and reached local min
        where average fitness over last n generations has not changed
        """
        if self.size <= self.stagnation_factor:
            return False
        else:
            last_gen2 = self.fitness_record[(self.generation - 2 - self.stagnation_factor):(self.generation - 2)]
            last_gen1 = self.fitness_record[(self.generation - 1 - self.stagnation_factor):(self.generation - 1)]
            return last_gen2 == last_gen1

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
            ['Min. fitness to survive:', self.min_fitness],
            ['Max. number of generations:', self.max_generations],
            ['Stagnation factor:', self.stagnation_factor],
            ['Minimize fitness:', self.minimiz_fitness],
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
                individual = Individual(random_tree(self.init_tree_size))
                self.individuals.append(individual)
                pb.animate(self.size)
        print '\n'
        self.describe_current()

    def roulette(self, number=None):
        """select parent pairs based on roulette method (probability proportional to fitness)"""
        number = number if number else self.size
        selections = []

        # unpack
        ranked_fitness, ranked_individuals = (list(i) for i in zip(*self.ranking))
        ranked_fitness = np.array(ranked_fitness)

        # calculate weighted probability proportial to fitness
        fitness_probability = ranked_fitness / float(ranked_fitness.sum())
        cum_prob_dist = fitness_probability.cumsum()

        for _ in xrange(number):
            # randomly select two individuals with weighted probability proportial to fitness
            p1 = ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])]
            p2 = ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])]
            selections.append((p1, p2))
        return selections

    def stochastic(self, number=None):
        """select parent pairs based on stochastic method (probability uniform across fitness)"""
        number = number if number else self.size
        selections = []

        # unpack
        ranked_fitness, ranked_individuals = (list(i) for i in zip(*self.ranking))
        ranked_fitness = np.array(ranked_fitness)

        # calculate weighted probability proportial to fitness
        fitness_probability = ranked_fitness / float(ranked_fitness.sum())
        cum_prob_dist = fitness_probability.cumsum()

        # determine uniform points
        p_dist = 1 / float(number)
        p0 = p_dist * r.random()
        points = p0 + p_dist * np.array(range(0, number))

        for p in points:
            # randomly select two individuals with weighted probability proportial to fitness
            p1 = ranked_individuals[bisect.bisect(cum_prob_dist, p * cum_prob_dist[-1])]
            p2 = ranked_individuals[bisect.bisect(cum_prob_dist, p * cum_prob_dist[-1])]
            selections.append((p1, p2))
        return selections

    def tournament(self, number=None, tournaments=4):
        """select parent pairs based on tournament method (random tournaments amoung individuals where fitness wins)"""
        number = number if number else self.size
        selections = []
        for _ in xrange(number):
            # select two groups of random competitors
            competitors1 = [self.ranking[i] for i in list(np.random.random_integers(0, self.size-1, tournaments))]
            competitors2 = [self.ranking[i] for i in list(np.random.random_integers(0, self.size-1, tournaments))]
            # groups compete in fitness tournament (local group sorting)
            competitors1.sort()
            competitors2.sort()
            # select most fit from each group
            winner1 = competitors1[0] if self.minimiz_fitness else competitors1[-1]
            winner2 = competitors2[0] if self.minimiz_fitness else competitors2[-1]
            selections.append((winner1[1], winner2[1]))
        return selections

    def rank_roulette(self, number=None, pressure=2):
        """select parent pairs based on rank roulette method (probability proportional to fitness rank)"""
        number = number if number else self.size
        selections = []

        # unpack
        _, ranked_individuals = (list(i) for i in zip(*self.ranking))

        # create a scaled rank by fitness (individuals already sorted, so just create rank range, then scale)
        n = self.size
        rank = range(n, 0, -1) if self.minimiz_fitness else range(1, n + 1)
        scaled_rank = 2.0 - pressure + (2.0 * (pressure - 1) * (np.array(rank) - 1) / (n - 1))

        # calculate weighted probability proportial to scaled rank
        scaled_rank_probability = scaled_rank / float(scaled_rank.sum())
        cum_prob_dist = scaled_rank_probability.cumsum()

        for _ in xrange(number):
            # randomly select two individuals with weighted probability proportial to scaled rank
            p1 = ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])]
            p2 = ranked_individuals[bisect.bisect(cum_prob_dist, r.random() * cum_prob_dist[-1])]
            selections.append((p1, p2))
        return selections

    def rank(self):
        """create ranking of individuals"""
        self.ranking = zip(self.fitness, self.individuals)
        self.ranking.sort()

    def create_generation(self):
        """create the next generations, this is main function that loops"""

        # determine fitness of current generations and log average fitness
        self.fitness_record[self.generation] = self.fitness.mean()

        # rank individuals by fitness
        self.rank()

        # selection parent chromosome pairs
        parent_pairs = self.roulette(int(self.size/4.0)) \
            + self.stochastic(int(self.size/4.0)) \
            + self.tournament(int(self.size/4.0)) \
            + self.rank_roulette(int(self.size/4.0), self.rank_pressure)

        # mate next generation
        next_generation = [p1.mate(p2) for p1, p2 in parent_pairs]

        # create new population
        self.individuals = next_generation

        # clear cached values
        self._fitness = np.array([])

        # log generation
        self.generation += 1

    def run(self, number_of_generations=None):
        """run algorithm"""

        number_of_generations = number_of_generations if number_of_generations else self.max_generations
        while self.generation < number_of_generations or not self.stagnate:
            self.create_generation()
