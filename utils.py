import re
import os
import random
import numpy as np
import itertools


def convert_var_to_chem(x):
    return x.upper()


def convert_chem_to_var(x):
    return x.lower()


def tuple_to_string(tuple):
    return ''.join([str(x) for x in tuple])


def np_matmul_str_matrices_2d(x, y):
    assert x.ndim == 2  # 2d matrices
    assert x.shape[1] == y.shape[0]  # #cols of x = #rows of y
    print(x, y)
    xrows = x.shape[0]
    cols = x.shape[1]
    ycols = y.shape[1]

    # Broadcasting
    if ycols == 1:
        y = np.repeat(y, xrows, axis=1)

    res = np_create_empty_string_array_2d(xrows, ycols)
    plus = np.core.defchararray.add('+', np_create_empty_string_array_2d(xrows,
                                                                         ycols))
    bar = np.core.defchararray.add('|', np_create_empty_string_array_2d(xrows,
                                                                         ycols))

    for k in range(cols):
        # Broadcasting bro. Look it up.
        emp = np.core.defchararray.add(x[:, k].reshape(cols, 1), bar)
        emp = np.core.defchararray.add(emp, y[k, :].reshape(1, cols))
        res = np.core.defchararray.add(res, emp)
        if k != cols - 1:
            res = np.core.defchararray.add(res, plus)
    return res


def np_create_empty_string_array_2d(nrows, ncols, default_str=''):
    l = [[default_str for j in range(ncols)] for i in range(nrows)]
    return np.array(l, dtype=str)


def convert_scalar_to_dual_rail(symbol=''):
    return {
        'pos': [symbol + 'p'],
        'neg': [symbol + 'n']
    }


def convert_expr_to_dual_rail(symbols=['a', 'b']):
    assert len(symbols) > 0
    ret = convert_scalar_to_dual_rail(symbols[0])
    for symbol in symbols[1:]:
        currvar = convert_scalar_to_dual_rail(symbol)
        ret = {
            'pos': meld(ret['pos'], currvar['pos']) + meld(ret['neg'],
                                                           currvar['neg']),
            'neg': meld(ret['pos'], currvar['neg']) + meld(ret['neg'],
                                                           currvar['pos'])
        }
    return ret


def meld(a=['a', 'b'], b=['c', 'd', 'e']):
    cartesian = itertools.product(a, b)
    ret = [x + '|' + y for x, y in cartesian]
    return ret


if __name__ == '__main__':
    print(convert_expr_to_dual_rail())