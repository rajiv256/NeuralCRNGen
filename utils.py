import src.ode as ode
import itertools
import re
import numpy as np
from src.reaction import Species
from src.ode import Scalar


def convert_dual_scalar_to_species(var: Scalar):
    varname = var.name
    spname = varname[0].upper() + varname[1:]  # z1p --> Z1p
    return Species(name=spname)


def convert_species_to_dual_scalar(sp: Species):
    spname = sp.name
    varname = spname[0].lower() + spname[1:]  # Z1p --> z1p
    return Scalar(name=varname)


def tuple_to_string(t):
    return ''.join([str(x) for x in t])


def np_matmul_str_matrices_2d(x, y):
    """
    Simulates the multiplication of two 2D matrices
    NOTE: Might contain trailing +++++
    :param x: 2d np array of type str
    :param y: 2d np array of type str
    :return: 2d np array of type str

    Example:
    x: np.array([['a1', 'a2']], dtype=str)
    y: np.array(
        [
            ['z1', 'z2', '0', '0'],
            ['0', '0', 'z1', 'z2']
        ]
    )
    result: [['a1|z1+' 'a1|z2+' 'a2|z1' 'a2|z2']]
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[0]

    xrows = x.shape[0]
    ycols = y.shape[1]
    cols = x.shape[1]

    res = np_create_empty_string_array_2d(xrows, ycols)
    plus = np.core.defchararray.add('+', np_create_empty_string_array_2d(xrows,
                                                                         ycols))
    bar = np.core.defchararray.add('|', np_create_empty_string_array_2d(xrows,
                                                                        ycols))
    plus_boolean = np.ones((xrows, ycols)) * True

    for k in range(cols):
        # New xrows, ycols matrix by repeating a col of x
        xtemp = np.repeat(x[:, k].reshape(-1, 1), ycols, axis=1)
        # New xrows, ycols matrix by repeating a row of y
        ytemp = np.repeat(y[k, :].reshape(1, -1), xrows, axis=0)

        prod = np.core.defchararray.add(xtemp, bar)
        prod = np.core.defchararray.add(prod, ytemp)

        # If one of the multiplicands is zero, prod is zero
        prod = np.where(ytemp == '0', '', prod)
        prod = np.where(xtemp == '0', '', prod)
        res = np.core.defchararray.add(res, prod)
        plus_updated = np.where(res == '', '', plus)
        plus_updated = np.where(prod == '', '', plus_updated)
        if k != cols - 1:
            res = np.core.defchararray.add(res, plus_updated)
    return res


def np_create_empty_string_array_2d(nrows, ncols, default_str=''):
    l = [[default_str for j in range(ncols)] for i in range(nrows)]
    return np.array(l, dtype=str)


def convert_scalar_to_dual_rail(scalar: Scalar):
    return {
        'pos': [Scalar(name=scalar.name + globals.POS_SUFFIX)],
        'neg': [Scalar(name=scalar.name + globals.NEG_SUFFIX)],
    }


def convert_expr_to_dual_rail(scalars=[Scalar()], parity=1):
    """
    Converts a polynomial expression of scalars into the dual-rail format
    based on the parity.
    :param scalars: scalar symbols
    :param parity: parity of -1 represents the negative sign. Anything else
    represents the poisitive sign
    :return: Returns
    """
    assert len(scalars) > 0
    ret = convert_scalar_to_dual_rail(scalars[0])
    for symbol in scalars[1:]:
        currscalar = convert_scalar_to_dual_rail(symbol)
        ret = {
            'pos': meld(ret['pos'], currscalar['pos']) + meld(ret['neg'],
                                                           currscalar['neg']),
            'neg': meld(ret['pos'], currscalar['neg']) + meld(ret['neg'],
                                                           currscalar['pos'])
        }

    # Reverse the parity
    if parity == -1:
        tmp = ret['pos']
        ret['pos'] = ret['neg']
        ret['neg'] = tmp

    return ret


def meld(a=['l', 'm'], b=['n', 'o', 'p']):
    """
    Calculates the cartesian product of strings by annealing the products
    together.
    """
    cartesian = itertools.product(a, b)
    ret = []
    for x, y in cartesian:
        # Handles mult with 0
        if x == '0' or y == '0':
            continue
        elif (x == '1' and y != '1') or (x != '1' and y == '1'):
            ret += [x]
        else:
            ret.append(x + '|' + y)
    return ret


def clean(s, sep=' '):
    s = s.strip()  # removes leading and trailing commas
    s = sep.join(s.split(sep))  # removes extra separators
    s = globals.SPACE.join(s.split(globals.SPACE))  # removes extra spaces


def print_crn(crn, title=''):
    print(title + '\n')
    ret = []
    for r in crn:
        print(r)
        ret += r.assign_concentrations()
    return ret