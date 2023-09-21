import itertools
import numpy as np
from src.reaction import Species
import src.algebra as algebra
from src.algebra import Scalar, Term, Expression


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


def np_create_empty_scalar_array(rows, cols, default=Scalar()):
    l = [[default for j in range(cols)] for i in range(rows)]
    return np.array(l)


def np_matmult_scalar_matrices_2d(x, y):
    """
    :param x: np array of type Scalar and by extension of type Term and
    Expression
    :param y: same as x
    :return: same as x and y
    """
    assert x.ndim == 2  # 2D matrix
    assert y.ndim == 2  # 2D matrix
    assert x.shape[1] == y.shape[0]
    xrows = x.shape[0]
    cols = x.shape[1]
    ycols = y.shape[1]

    # Initialize ret
    ret = []
    for xrow in range(xrows):
        ret.append([])
        for ycol in range(ycols):
            ret[xrow].append(Expression())

    for xrow in range(xrows):
        for ycol in range(ycols):
            for col in range(cols):
                ret[xrow][ycol] = algebra.add_expressions(
                    ret[xrow][ycol],
                    algebra.mult_expressions(x[xrow][col], y[col][ycol])
                )
    ret = np.array(ret, dtype=Expression)
    return ret


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


def clean(s, sep=' '):
    s = s.strip()  # removes leading and trailing commas
    s = sep.join(s.split(sep))  # removes extra separators
    s = globals.SPACE.join(s.split(globals.SPACE))  # removes extra spaces


def print_crn(crn, title=''):
    ret = []
    for r in crn:
        ret += r.assign_concentrations()
        print(r)
    return ret


def print_concentrations(cs, items_per_row=2):
    cs.sort()
    print("[")  # Here for formatting,  don't delete.
    iter = 0
    while iter < len(cs):
        print(', '.join(cs[iter:iter+items_per_row]))
        iter += items_per_row
    print("]")


def print_doubly_nested_list(l):
    for x in l:
        print('; '.join([str(y) for y in x]))