# Fast-Cubic-Spline-Python provides an implementation of 1D and 2D fast spline
# interpolation algorithm (Habermann and Kindermann 2007) in Python.
# Copyright (C) 2012, 2013 Joon H. Ro
'''
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
'''
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs, fmin
from cython.parallel import prange

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.

'''
Cubic spline interpolation using Habermann and Kindermann (2007)'s algorithm 
'''
#----------------------------------------------------------------------
cdef double Pi(double t) nogil:
    cdef:
        double abs_t = fabs(t)
    if abs_t <= 1:
        return(4 - 6 * abs_t**2 + 3 * abs_t **3)
    elif abs_t <= 2:
        return((2 - abs_t)**3)
    else:
        return(0)

#----------------------------------------------------------------------        
cdef double u(double x, int k, double a, double h) nogil:
    return(Pi((x - a)/h - (k - 2)))

#----------------------------------------------------------------------
def interpolate(double x,
                double a, double b,
                double[:] c,
                ):
    '''
    Return interpolated function value at x
    
    Parameters
    ----------
    x : float
        The value where the function will be approximated at
    a : double
        Lower bound of the grid
    b : double
        Upper bound of the grid
    c : ndarray
        Coefficients of spline
    
    Returns
    -------
    out : float
        Approximated function value at x
    '''
    return(_interpolate(x, a, b, c))

cdef double _interpolate(double x,
                         double a, double b,
                         double[:] c,
                         ) nogil:

    cdef:
        int n = c.shape[0] - 3
        double h = (b - a)/n
        int l = <int>((x - a)//h) + 1
        int m = <int>(fmin(l + 3, n + 3))
        int i1
        double s = 0

    for i1 in xrange(l, m + 1):
        s += c[i1 - 1] * u(x, i1, a, h)

    return(s)

#----------------------------------------------------------------------
def interpolate_2d(double x, double y,
                   double a1, double b1,
                   double a2, double b2,
                   double[:, :] c,
                   ):
    '''
    Return interpolated function value at x
    
    Parameters
    ----------
    x, y : float
        The values where the function will be approximated at
    a1, b1 : double
        Lower and upper bounds of the grid for x
    a2, b2 : double
        Lower and upper bounds of the grid for y
    c : ndarray
        Coefficients of spline
    
    Returns
    -------
    out : float
        Approximated function value at (x, y)
    '''
    return(_interpolate_2d(x, y, a1, b1, a2, b2, c))
    
cdef double _interpolate_2d(double x, double y,
                            double a1, double b1,
                            double a2, double b2,
                            double[:, :] c,
                            ) nogil:
    cdef:
        int n1 = c.shape[0] - 3
        int n2 = c.shape[1] - 3
        double h1 = (b1 - a1)/n1
        double h2 = (b2 - a2)/n2
        int l1 = <int>((x - a1)//h1) + 1
        int l2 = <int>((y - a2)//h2) + 1
        int m1 = <int>(fmin(l1 + 3, n1 + 3))
        int m2 = <int>(fmin(l2 + 3, n2 + 3))
        int i1, i2
        double s = 0
        double u_x, u_y

    for i1 in xrange(l1, m1 + 1):
        u_x = u(x, i1, a1, h1)
        for i2 in xrange(l2, m2 + 1):
            u_y = u(y, i2, a2, h2)
            s += c[i1 - 1, i2 - 1] * u_x * u_y

    return(s)


# # note: function also modifies b[] and d[] params while solving
# cpdef np.ndarray TDMASolve(double[:] a, double[:] b, double[:] c, double[:] d, unsigned int n):
#     cdef:
#         unsigned int ii
#         np.ndarray[np.double_t] xNumpy = np.empty(n)
#         double* x = &xNumpy[0]
#     # n is the numbers of rows, a and c has length n-1
#     for ii in range(n-1):
#         d[ii+1] -= d[ii]*a[ii]/b[ii]
#         b[ii+1] -= c[ii]*a[ii]/b[ii]
#     for ii in range(n-1):#range(n-2,-1,-1):       
#         ii = n-2-ii
#         d[ii] -= d[ii+1]*c[ii]/b[ii+1]
#     for ii in range(n):
#         x[ii] = d[ii]/b[ii]
#     return xNumpy

# note: function also modifies b[] and d[] params while solving
def TDMASolve(a, b, c, d):
    n = len(d) # n is the numbers of rows, a and c has length n-1
    for i in xrange(n-1):
        d[i+1] -= d[i] * a[i] / b[i]
        b[i+1] -= c[i] * a[i] / b[i]
    for i in reversed(xrange(n-1)):
        d[i] -= d[i+1] * c[i] / b[i+1]
    return [d[i] / b[i] for i in xrange(n)] # return the solution