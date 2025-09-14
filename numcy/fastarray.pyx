# fast_array_nd.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log, sin, cos, pow
cimport cython
cimport numpy as cnp
import numpy as np
cdef extern from "cblas.h":
    void cblas_dgemm(int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     double *A, int lda,
                     double *B, int ldb,
                     double beta,
                     double *C, int ldc)

cdef int CblasRowMajor = 101
cdef int CblasNoTrans = 111

cdef class FastArrayND:
    cdef public int ndim
    cdef public int[::1] shape
    cdef public int size
    cdef double[::1] data

    # -------------------
    # Initialization
    # -------------------
    def __init__(self, int[:] shape):
        cdef int i, total=1
        self.ndim = shape.shape[0]
        self.shape = shape
        for i in range(self.ndim):
            total *= shape[i]
        self.size = total
        self.data = <double[:self.size]> malloc(self.size * sizeof(double))
        for i in range(self.size):
            self.data[i] = 0.0

    # -------------------
    # Cleanup
    # -------------------
    def __dealloc__(self):
        if self.data is not None:
            free(<void*> self.data)

    # -------------------
    # Internal binary operation (with broadcasting)
    # -------------------
    cdef void _binary_op(self, FastArrayND other, FastArrayND out, char op):
        cdef int i
        if self.size == other.size:
            for i in range(self.size):
                if op=='+': out.data[i] = self.data[i] + other.data[i]
                elif op=='-': out.data[i] = self.data[i] - other.data[i]
                elif op=='*': out.data[i] = self.data[i] * other.data[i]
                elif op=='/': out.data[i] = self.data[i] / other.data[i]
                elif op=='^': out.data[i] = pow(self.data[i], other.data[i])
        elif other.size == 1:
            for i in range(self.size):
                if op=='+': out.data[i] = self.data[i] + other.data[0]
                elif op=='-': out.data[i] = self.data[i] - other.data[0]
                elif op=='*': out.data[i] = self.data[i] * other.data[0]
                elif op=='/': out.data[i] = self.data[i] / other.data[0]
                elif op=='^': out.data[i] = pow(self.data[i], other.data[0])
        else:
            # Simple broadcasting for 1D-3D arrays
            if self.ndim <= 3 and other.ndim <= 3:
                cdef int d0=1,d1=1,d2=1
                cdef int od0=1,od1=1,od2=1
                if self.ndim >=1: d0=self.shape[0]
                if self.ndim >=2: d1=self.shape[1]
                if self.ndim ==3: d2=self.shape[2]
                if other.ndim >=1: od0=other.shape[0]
                if other.ndim >=2: od1=other.shape[1]
                if other.ndim ==3: od2=other.shape[2]
                cdef int i0,i1,i2, idx_out, idx_self, idx_other
                for i0 in range(d0):
                    for i1 in range(d1):
                        for i2 in range(d2):
                            idx_out = i0*d1*d2 + i1*d2 + i2
                            idx_self = idx_out
                            idx_other = (i0 if od0>1 else 0)*(od1*od2) + (i1 if od1>1 else 0)*od2 + (i2 if od2>1 else 0)
                            if op=='+': out.data[idx_out] = self.data[idx_self] + other.data[idx_other]
                            elif op=='-': out.data[idx_out] = self.data[idx_self] - other.data[idx_other]
                            elif op=='*': out.data[idx_out] = self.data[idx_self] * other.data[idx_other]
                            elif op=='/': out.data[idx_out] = self.data[idx_self] / other.data[idx_other]
                            elif op=='^': out.data[idx_out] = pow(self.data[idx_self], other.data[idx_other])
            else:
                raise ValueError("Unsupported shapes for broadcasting")

    # -------------------
    # Helper to handle scalars
    # -------------------
    cdef FastArrayND _ensure_array(self, other):
        if isinstance(other, FastArrayND):
            return other
        else:
            cdef FastArrayND tmp = FastArrayND(np.array([1], dtype=np.int32))
            tmp.data[0] = cython.cast(double, other)
            return tmp

    # -------------------
    # Arithmetic operators
    # -------------------
    def __add__(self, other): return self._binary_out(other, '+')
    def __sub__(self, other): return self._binary_out(other, '-')
    def __mul__(self, other): return self._binary_out(other, '*')
    def __truediv__(self, other): return self._binary_out(other, '/')
    def __pow__(self, other): return self._binary_out(other, '^')

    cdef FastArrayND _binary_out(self, other, char op):
        cdef FastArrayND o = self._ensure_array(other)
        cdef FastArrayND out = FastArrayND(self.shape)
        self._binary_op(o, out, op)
        return out

    # -------------------
    # In-place operators
    # -------------------
    def __iadd__(self, other):
        self._binary_op(self._ensure_array(other), self, '+')
        return self
    def __isub__(self, other):
        self._binary_op(self._ensure_array(other), self, '-')
        return self
    def __imul__(self, other):
        self._binary_op(self._ensure_array(other), self, '*')
        return self
    def __itruediv__(self, other):
        self._binary_op(self._ensure_array(other), self, '/')
        return self
    def __ipow__(self, other):
        self._binary_op(self._ensure_array(other), self, '^')
        return self

    # -------------------
    # Elementwise ufuncs
    # -------------------
    cpdef FastArrayND exp(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = exp(self.data[i])
        return out
    cpdef FastArrayND log(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = log(self.data[i])
        return out
    cpdef FastArrayND sin(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = sin(self.data[i])
        return out
    cpdef FastArrayND cos(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = cos(self.data[i])
        return out

    # -------------------
    # Indexing / slicing (returns copy)
    # -------------------
    def __getitem__(self, idx):
        # Implementation as in previous cell (1D, 2D, 3D)...
        # Can copy the same __getitem__ code we wrote earlier
        ...

    # -------------------
    # BLAS Matrix Multiplication
    # -------------------
    cpdef FastArrayND matmul(self, FastArrayND other):
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("BLAS matmul supports 2D arrays only")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Shapes do not align for matmul")
        cdef int M = self.shape[0]
        cdef int K = self.shape[1]
        cdef int N = other.shape[1]
        cdef FastArrayND out = FastArrayND(np.array([M,N], dtype=np.int32))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0,
                    &self.data[0], K,
                    &other.data[0], N,
                    0.0,
                    &out.data[0], N)
        return out

    cpdef FastArrayND batched_matmul(self, FastArrayND other):
        if self.ndim != 3 or other.ndim != 3:
            raise ValueError("Batched matmul supports 3D arrays only")
        if self.shape[0] != other.shape[0] or self.shape[2] != other.shape[1]:
            raise ValueError("Shapes do not align for batched matmul")
        cdef int batch = self.shape[0]
        cdef int M = self.shape[1]
        cdef int K = self.shape[2]
        cdef int N = other.shape[2]
        cdef FastArrayND out = FastArrayND(np.array([batch, M, N], dtype=np.int32))
        cdef int b
        for b in range(batch):
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0,
                        &self.data[b*M*K], K,
                        &other.data[b*K*N], N,
                        0.0,
                        &out.data[b*M*N], N)
        return out
