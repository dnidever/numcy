# tests/test_fast_array.py
import numpy as np
from numcy import FastArrayND

def test_addition():
    a = FastArrayND(np.array([2,2], dtype=np.int32))
    b = FastArrayND(np.array([2,2], dtype=np.int32))
    for i in range(4):
        a.data[i] = i
        b.data[i] = 10*i
    c = a + b
    for i in range(4):
        assert c.data[i] == a.data[i] + b.data[i]

