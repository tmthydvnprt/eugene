# pylint: disable=invalid-name
"""
Primatives.py
"""

from __future__ import division

import random as r
import operator as o
import numpy as np
import scipy.misc as sm
import scipy.special as sp

np.seterr(all='ignore')

# + - * / %
BINARIES = ['n_add', 'n_sub', 'n_mul', 'n_div', 'n_mod', 'n_pow']
# asb, sqtr, log, exp, sin, cos, tan, min, max,
UNARIES = ['n_abs', 'n_sqrt', 'n_log', 'n_exp', 'n_sin', 'n_cos', 'n_tan', 'n_min', 'n_max']
# none
NARIES = ['']
# 3.14..., 2.71...
CONSTS = ['np.pi', 'np.e']
# random constants

def random_int(rmin=-500, rmax=500):
    """Random intgeter with custom defaults"""
    return r.randint(rmin, rmax)

def random_uniform(rmin=-500, rmax=500):
    """Random uniform with custom defaults"""
    return r.randint(rmin, rmax)

def random_normal(rmean=0, rscale=100):
    """Random uniform with custom defaults"""
    return r.normalvariate(rmean, rscale)

EPHEMERAL = {
    0: random_int,
    1: r.random,
    2: random_uniform,
    3: random_normal
}
# UNARIES = [
#     'n_abs', 'n_inv', 'n_neg', 'n_pos', 'n_acos', 'n_acosh', 'n_asin', 'n_asinh', 'n_atan', 'n_atanh', 'n_ceil', 'n_cos', \
#     'n_cosh', 'n_degrees', 'n_exp', 'n_expm1', 'n_fabs', 'n_factorial', 'n_floor', 'n_gamma', 'n_isinf', \
#     'n_isnan', 'n_gammaln', 'n_log10', 'n_log2', 'n_log1p', 'n_log', 'n_radians', 'n_sin', 'n_sinh', 'n_sqrt', 'n_tan', \
#     'n_tanh', 'n_trunc'
# ]
# BINARIES = [
#     'n_or', 'n_add', 'n_and', 'n_div', 'n_eq', 'n_floordiv', 'n_ge', 'n_gt', 'n_le', 'n_lt', 'n_mod', 'n_mul', \
#     'n_ne', 'n_sub', 'n_xor', 'n_atan2', 'n_copysign', 'n_fmod', 'n_hypot', 'n_ldexp', 'n_pow', 'n_round'
# ]
# NARIES = ['n_max', 'n_min', 'n_sum', 'n_prod']

n_abs = o.abs
n_neg = o.neg
n_pos = o.pos
n_eq = o.eq
n_ge = o.ge
n_gt = o.gt
n_le = o.le
n_lt = o.lt
n_mul = o.mul
n_ne = o.ne
n_add = o.add
n_sub = o.sub
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

# safe functions, graceful error fallbacks
def intify(a):
    """safe intify"""
    # check if a = 3.14, np.array(3.14), or np.array([3.14])
    if np.isscalar(a) or a.shape == () or a.shape == (1,):
        return int(a) if not np.isnan(a) and np.isfinite(a) else 1
    # must be a 'real' array
    else:
        return np.array([int(_) if not np.isnan(_) and np.isfinite(_) else 1 for _ in a])

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
