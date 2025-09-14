# utils.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport sqrt, exp, log
cimport cython
import numpy as np
cimport numpy as np

# -------------------
# Basic arithmetic helpers
# -------------------
cpdef double add(double a, double b):
    """Return a + b"""
    return a + b

cpdef double sub(double a, double b):
    """Return a - b"""
    return a - b

cpdef double mul(double a, double b):
    """Return a * b"""
    return a * b

cpdef double div(double a, double b):
    """Return a / b"""
    return a / b

# -------------------
# Array helpers
# -------------------
cpdef double[:] add1d(double[:] arr1, double[:] arr2):
    """Add two 1D arrays elementwise"""
    cdef Py_ssize_t n = arr1.shape[0]
    cdef Py_ssize_t i
    cdef double[:] out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = arr1[i] + arr2[i]
    return out

