"""
Eugene
"""

__all__ = ["Primatives", "Node", "Tree", "List", "String", "Individual", "Population", "Util"]
__version__ = '0.1.1'
__date__ = '2015-08-07 07:09:00 -0700'
__author__ = 'tmthydvnprt'
__status__ = 'development'
__website__ = 'https://github.com/tmthydvnprt/eugene'
__email__ = 'tmthydvnprt@users.noreply.github.com'
__maintainer__ = 'tmthydvnprt'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015, eugene'
__credits__ = ''

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
