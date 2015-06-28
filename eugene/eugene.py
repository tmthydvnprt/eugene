"""
Eugene

Operators :
===========
    Unary :
    -------
        o.abs(a)          - Same as abs(a).
        o.inv(a)          - Same as ~a.
        o.neg(a)          - Same as -a.
        o.pos(a)          - Same as +a.

    Binary :
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

Functions :
===========
    Unary :
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

    Binary :
    --------
        np.atan2(y, x)     - Return the arc tangent (measured in radians) of y/x. Unlike atan(y/x), the signs of both x and y are considered.
        np.copysign(x, y)  - Return x with the sign of y.
        np.fmod(x, y)      - Return fmod(x, y), according to platform C.  x % y may differ.
        np.hypot(x, y)     - Return the Euclidean distance, sqrt(x*x + y*y).
        np.ldexp(x, i)     - Return x * (2**i).
        np.pow(x, y)       - Return x**y (x to the power of y).
        np.round(x[, y])   - Return the floating point value x rounded to y digits after the decimal point.

    N-ary :
    --------
        np.max(x, y, ...)  - Return the largest item in an iterable or the largest of two or more arguments.
        np.min(x, y, ...)  - Return the smallest item in an iterable or the smallest of two or more arguments.
        np.fsum([x,y,...]) - Return an accurate floating point sum of values in the iterable.
        np.prod([x,y,...]) - Return an accurate floating point product of values in the iterable.

Constants :
===========
    np.p - The mathematical constant pi = 3.141592..., to available precision.
    np.e - The mathematical constant e  = 2.718281..., to available precision.

Ephemeral Variables : - once created stay constant, only can be used during initialization or mutation
=====================
    r.random()                 - Returns x in the interval [0, 1).
    r.randint(a, b)            - Returns integer x in range [a, b].
    r.uniform(a, b)            - Returns number x in the range [a, b) or [a, b] depending on rounding.
    r.normalvariate(mu, sigma) - Returns number x from Normal distribution. mu is the mean, and sigma is the standard deviation.

Variables :
===========
    a, b, c, ..., x, y, z    - whatever you need

To add :
========
pd.shift()

removed :
=========
o.not_() - can't operate on array, but o.inv() can & produces the same result

"""

# dependancies
import operator      as o
import random        as r
import numpy         as np
import pandas        as pd
import copy          as cp
import scipy.special as sp
import scipy.misc    as sm

variables = ['x']

unaries   = [
    'n_abs', 'n_inv', 'n_neg', 'n_pos', 'n_acos', 'n_acosh', 'n_asin', 'n_asinh', 'n_atan', 'n_atanh', 'n_ceil', 'n_cos', \
    'n_cosh', 'n_degrees', 'n_erf', 'n_erfc', 'n_exp', 'n_expm1', 'n_fabs', 'n_factorial', 'n_floor', 'n_gamma', 'n_isinf', \
    'n_isnan','n_gammaln', 'n_log10', 'n_log2', 'n_log1p', 'n_log', 'n_radians', 'n_sin', 'n_sinh', 'n_sqrt', 'n_tan', \
    'n_tanh', 'n_trunc'
]
binaries  = [
    'n_or', 'n_add', 'n_and', 'n_div', 'n_eq', 'n_floordiv', 'n_ge', 'n_gt', 'n_le', 'n_lt', 'n_mod', 'n_mul', \
    'n_ne', 'n_sub', 'n_xor', 'n_atan2', 'n_copysign', 'n_fmod', 'n_hypot', 'n_ldexp', 'n_pow', 'n_round'
]
naries    = ['n_max', 'n_min', 'n_sum', 'n_prod']
consts    = ['np.pi', 'np.e']

ephemeral = {
    0: r.randint(-500,500),
    1: r.random(),
    2: r.uniform(-500,500),
    3: r.normalvariate(0,100)
}

n_abs       =       o.abs

n_neg       =       o.neg
n_pos       =       o.pos
n_or        =       o.or_
n_add       =       o.add
n_and       =       o.and_

n_eq        =       o.eq

n_ge        =       o.ge
n_gt        =       o.gt
n_le        =       o.le
n_lt        =       o.lt

n_mul       =       o.mul
n_ne        =       o.ne
n_sub       =       o.sub
n_xor       =       o.xor
n_acos      =       np.arccos
n_acosh     =       np.arccosh
n_asin      =       np.arcsin
n_asinh     =       np.arcsinh
n_atan      =       np.arctan
n_atanh     =       np.arctanh
n_ceil      =       np.ceil
n_cos       =       np.cos
n_cosh      =       np.cosh
n_degrees   =       np.degrees
n_erf       =       sp.erf
n_erfc      =       sp.erfc
n_exp       =       np.exp
n_expm1     =       np.expm1
n_fabs      =       np.fabs
n_factorial =       sm.factorial
n_floor     =       np.floor
n_gamma     =       sp.gamma
n_isinf     =       np.isinf
n_isnan     =       np.isnan
n_gammaln   =       sp.gammaln
n_log10     =       np.log10
n_log2      =       np.log2
n_log1p     =       np.log1p
n_radians   =       np.radians
n_sin       =       np.sin
n_sinh      =       np.sinh
n_sqrt      =       np.sqrt
n_tan       =       np.tan
n_tanh      =       np.tanh
n_trunc     =       np.trunc
n_atan2     =       np.arctan2
n_copysign  =       np.copysign
n_fmod      =       np.fmod
n_hypot     =       np.hypot
n_ldexp     =       np.ldexp
n_log       =       np.log
n_pow       =       np.power
n_round     =       np.around

# safe functions, graceful error fallbacks
def intify(x):
    return 1 if np.isnan(x) or not np.isfinite(x) else int(x) if type(x) != pd.core.series.Series else x.fillna(0).astype(int)

def n_inv(a):
    return o.inv(intify(a))

def n_and(a, b):
    return o.and_(intify(a), intify(b))

def n_or(a, b):
    return o.or_(intify(a), intify(b))

def n_xor(a, b):
    return o.xor(intify(a), intify(b))

def n_mod(a, b):
    return pd.Series(np.where(b != 0, o.mod(a, b), 1))

def n_div(a, b):
    return pd.Series(np.where(b != 0, o.div(a, b), 1))

def n_floordiv(a, b):
    return pd.Series(np.where(b != 0, o.floordiv(a, b), 1))

def n_round(a, b):
    element_round = np.vectorize(np.round)
    return element_round(a, intify(b))

def n_ldexp(a, b):
    return np.ldexp(a, intify(b))

# n-ary custom functions
def n_max(x):
    return reduce(np.maximum, x)
def n_min(x):
    return reduce(np.minimum, x)
def n_sum(x):
    return reduce(np.add, x)
def n_prod(x):
    return reduce(np.multiply, x)

class Node(object):
    """
    Defines a node of the tree
    """

    def __init__(self, value=None, *children):
        self.value    = value
        self.children = children
        self.num      = None
        self.total    = None

    def __repr__(self):
         return self.__str__()

    def __str__(self):
        # node is a variable or constant
        if len(self.children) == 0:
            return str(self.value)
        # node is a unary, binary or n-ary function
        else:
            if self.value in naries:
                return str(self.value) + '([' + ','.join([str(c) for c in self.children]) + '])'
            else :
                return str(self.value) + '(' + ','.join([str(c) for c in self.children]) + ')'

class Tree(object):
    """
    Defined a Tree with nodes
    """

    def __init__(self, nodes=None):
        self.nodes = nodes

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.nodes)

    def evaluate(self):
        try:
            return eval(compile(self.__str__(), '', 'eval'))
        except:
            return pd.Series(np.zeros(x.shape))

    def set_nums(self, count=-1) :
        count += 1
        self.nodes.num = count
        if len(self.nodes.children) > 0 :
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
        if self.nodes.num == n :
            return cp.deepcopy(self.nodes)
        elif len(self.nodes.children) > 0 :
            for c in (self.nodes.children) :
                cn = Tree(c).get_node(n)
                if cn :
                    return cn
        else :
            return None

    def set_node(self, n=0, node=None):
        """set a node in the tree"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()
        # search tree until node number is found, and store sub tree
        if self.nodes.num == n:
            self.nodes = node
        else :
            self.nodes.children = tuple([Tree(c).set_node(n, node) for c in self.nodes.children])
        return self.nodes

    def display(self, level=0, level_list=None):
        """display helper"""
        level_list = level_list if level_list else []
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0 or level == 0:
            self.set_nums()
        if level == 0:
            nodeStr = '[0:' + str(self.nodes.num) + '] ' + str(self.nodes.value)
        else:
            if level_list[-1] == '      ':
                nodeStr = '    ' + ''.join(level_list[:-1]) + '\-[' + str(level) +':'+ str(self.nodes.num) +'] '+ str(self.nodes.value)
            else :
                nodeStr = '    ' + ''.join(level_list[:-1]) + '|-[' + str(level) +':'+ str(self.nodes.num) +'] '+ str(self.nodes.value)
        print nodeStr
        for i, child in enumerate(self.nodes.children):
            Tree(child).display(level+1, level_list + ['      ' if i == len(self.nodes.children) - 1 else '|     '])

    def edges(self):
        """get edges of tree"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()
        # get list of tuple edges between nodes e.g. [(n1,n2),(n1,n3)...]
        edges         = [(self.nodes.value + str(self.nodes.num), c.value + str(c.num) if len(c.children) > 0 else c.value) for c in self.nodes.children]
        childrenNodes = [Tree(c).edges() for c in self.nodes.children if len(c.children) > 0]
        for i in range( len( childrenNodes ) ) :
            edges += childrenNodes[i]
        return edges

    def nodes(self):
        """return nodes of tree"""
        # fill node numbers if blank or if on root node
        if self.nodes.num == None or self.nodes.num == 0:
            self.set_nums()
        # get list of nodes
        currentNode = [str(self.value) + str(self.num)]
        return currentNode + [c.value for c in self.children if len(c.children) == 0] + [c.nodes() for c in self.children if len(c.children) > 0]

def random_tree(max_level=20, min_level=1, current_level=0):
    """generate a random tree"""
    return Tree(random_node(max_level, min_level, current_level))

def random_node(max_level=20, min_level=1, current_level=0):
    """return a random node or nodes"""
    if current_level == max_level:
        rand_node = r.randint(0, 3)
        # return a constant
        if rand_node == 0:
            return Node(consts[r.randint(0, len(consts) - 1)])
        # return ephemeral constant random ( 0:1, uniform -500:500, or normal -500:500 )
        elif rand_node == 1:
            return Node(ephemeral[r.randint(1, len(ephemeral) - 1)])
        # return ephemeral constant random integer
        elif rand_node == 2:
            return Node(ephemeral[0])
        # return variable
        elif rand_node == 3:
            return Node(variables[r.randint(0, len(variables) - 1)])
    else:
        rand_node = r.randint(4, 6) if current_level < min_level else r.randint(0, 6)
        # return a constant
        if rand_node == 0:
            return Node(consts[r.randint(0, len(consts) - 1)])
        # return ephemeral constant random ( 0:1, uniform -500:500, or normal -500:500 )
        elif rand_node == 1:
            return Node(ephemeral[r.randint(1, len(ephemeral) - 1)])
        # return ephemeral constant random integer
        elif rand_node == 2:
            return Node(ephemeral[0])
        # return variable
        elif rand_node == 3:
            return Node(variables[r.randint(0, len(variables) - 1)])
        # return a unary operator
        elif rand_node == 4:
            return Node(unaries[r.randint(0,len(unaries) - 1)], random_node(max_level, min_level, current_level + 1))
        # return a binary operator
        elif rand_node == 5 :
            return Node(binaries[r.randint(0,len(binaries) - 1)], random_node(max_level, min_level, current_level + 1), random_node(max_level, min_level, current_level + 1))
        # return a n-ary operator
        elif rand_node == 6:
            naryNodeNum = r.randint(2, 5)
            if naryNodeNum == 2:
                return Node(naries[r.randint(0, len(naries) - 1)], random_node(max_level - 1, current_level + 1 ), random_node(max_level - 1, current_level + 1))
            elif naryNodeNum == 3:
                return Node(naries[r.randint(0, len(naries) - 1)], random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1))
            elif naryNodeNum == 4:
                return Node(naries[r.randint(0, len(naries) - 1)], random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1))
            elif naryNodeNum == 5:
                return Node(naries[r.randint(0, len(naries) - 1)], random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1), random_node(max_level - 1, current_level + 1))


def DEFAULT_OBJECTIVE(x):
    return 1.0

class Individual(object):
    """docstring for Individual"""

    def __init__(self, chromosomes = None):
        super(Individual, self).__init__()
        self.chromosomes = chromosomes

    def __repr__( self ):
        return self.__str__()

    def __str__( self ):
        return str( self.chromosomes )

    def fitness( self, objective_function=DEFAULT_OBJECTIVE):
        """return the fitness of an Individual based on the objective_function"""
        gene_expression = self.chromosomes.evaluate()
        return objective_function(gene_expression)

    def mate(self, spouse=None):
        """mate this Individual with a spouse"""
        pass

class Population(object):

    def __init__(self, size=100, ind_type='tree', max_size=4, min_size=2, gamma=1.0):
        self.size = size
        self.indType = ind_type
        self.max_size = max_size
        self.min_size = min_size
        self.created = False
        self.fitness = 0
        self.gamma = gamma

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.created:
            population_string  = 'Population\n'
            population_string += 'Size            : ' + str(self.size) + '\n'
            population_string += 'Average fitness : ' + str(self.fitness) + '\n'
            population_string += 'Max fitness     : ' + str(max(self.fitness)) + '\n'
            population_string += 'Min fitness     : ' + str(min(self.fitness)) + '\n'
            return str(population_string)
        else :
            population_string  = 'POPULATION NOT INITIALIZED\n'
            population_string += 'Size            : ' + str(self.size) + '\n'
            population_string += 'Average fitness : ' + str(self.fitness) + '\n'
            population_string += 'Max fitness     : ' + str(self.fitness) + '\n'
            population_string += 'Min fitness     : ' + str(self.fitness) + '\n'
            return str(population_string)

    def initialize(self, seed=None):
        """initialize a population based on seed or randomly"""
        self.created = True
        if seed :
            self.individuals = seed
            self.size = len(seed)
        else :
            self.randomTreePop()
