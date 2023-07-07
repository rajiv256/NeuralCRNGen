import re
import os
import random
import numpy as np


def convert_var_to_chem(x):
    return x.upper()


def convert_chem_to_var(x):
    return x.lower()


def tuple_to_string(tuple):
    return ''.join([str(x) for x in tuple])


def np_matmul_str_matrices_2d(x, y):
    assert x.ndim == 2
    assert x.shape[1] == y.shape[0] # #cols of x = #rows of y

    xrows = x.shape[0]
    cols = x.shape[1]
    ycols = y.shape[1]
    res = np_create_empty_string_array_2d(xrows, ycols)
    plus = np.core.defchararray.add('+', np_create_empty_string_array_2d(xrows, ycols))

    for k in range(cols):
        # Broadcasting bro. Look it up.
        emp = np.core.defchararray.add(x[:, k].reshape(cols, 1),
                                       y[k,:].reshape(1, cols))
        res = np.core.defchararray.add(res, emp)
        if k!=cols-1:
            res = np.core.defchararray.add(res, plus)
    return res


def np_create_empty_string_array_2d(nrows, ncols, default_str=''):
    l = [[default_str for j in range(ncols)] for i in range(nrows)]
    return np.array(l, dtype=str)