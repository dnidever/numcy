# fast_array_nd.pxd

cdef class FastArrayND:
    cdef public int ndim
    cdef public int[::1] shape
    cdef public int size
    cdef double[::1] data

    # -------------------
    # Constructor / Destructor
    # -------------------
    def __init__(self, int[:] shape)
    def __dealloc__(self)

    # -------------------
    # Arithmetic operators
    # -------------------
    def __add__(self, other)
    def __sub__(self, other)
    def __mul__(self, other)
    def __truediv__(self, other)
    def __pow__(self, other)

    def __iadd__(self, other)
    def __isub__(self, other)
    def __imul__(self, other)
    def __itruediv__(self, other)
    def __ipow__(self, other)

    # -------------------
    # Elementwise ufuncs
    # -------------------
    cpdef FastArrayND exp(self)
    cpdef FastArrayND log(self)
    cpdef FastArrayND sin(self)
    cpdef FastArrayND cos(self)

    # -------------------
    # Indexing / slicing
    # -------------------
    def __getitem__(self, idx)

    # -------------------
    # BLAS Matrix Multiplication
    # -------------------
    cpdef FastArrayND matmul(self, FastArrayND other)
    cpdef FastArrayND batched_matmul(self, FastArrayND other)
