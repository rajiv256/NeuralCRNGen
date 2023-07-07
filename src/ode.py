import itertools
import utils
from crn import CRN
import numpy as np


class MonomialODE:
    def __init__(self, lhs='', rhs=[]):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"""d{self.lhs}/dt = {''.join(self.rhs)} """

    def _to_catalytic_crn(self):
        """
        :rtype: CRN
        """
        crn = CRN(
            reactants=[utils.convert_var_to_chem(x) for x in self.rhs],
            products=[utils.convert_var_to_chem(x) for x in [self.lhs] +
                      self.rhs],
            rate_constant='1.0',
            reversible=False
        )
        return crn


class MultinomialODE:
    def __init__(self, monomials=[]):
        assert len(monomials) > 0
        assert sum([m.lhs != monomials[0].lhs for m in monomials]) == 0
        self.lhs = monomials[0].lhs
        self.multi_rhs = [m.rhs for m in monomials]
        self.monomials = monomials

    def __str__(self):
        return f"""d({self.lhs}/dt) = {'+'.join([''.join(rhs) for rhs in self.multi_rhs])}"""

    def _to_catalytic_crn(self):
        crns: list[CRN] = [m._to_catalytic_crn() for m in self.monomials]
        return crns


class Variable:
    def __init__(self, symbol='', dimensions=()):
        self.symbol = symbol
        self.dims = dimensions

    def __str__(self):
        print(self.symbol)

    def _vectorize(self):
        print(self.dims)
        dim_indices = [list(range(1, d+1)) for d in self.dims]
        concatenated_dims = itertools.product(*dim_indices)
        final_dims = np.array(
            [utils.tuple_to_string(t) for t in concatenated_dims],
            dtype=str
        ).reshape(self.dims)
        vect = np.core.defchararray.add(self.symbol, final_dims)
        return vect

    def __str__(self):
        var = self._vectorize()
        return str(var)


class ODESystem:
    def __init__(self, multinomials=[]):
        """Takes in a set of multinomials and saves them in a compressed
        matrix form."""
        self.multinomials = multinomials


if __name__ == '__main__':
    m = MonomialODE('x1p',
                    ['y1p', 'z1n'])
    m2 = MonomialODE('x1p', ['y1n', 'z1p'])
    print(m._to_catalytic_crn())
    mm = MultinomialODE([m, m2])
    for crn in mm._to_catalytic_crn():
        print(crn)
    for x in list(itertools.product(*[[1, 2], [3, 4]])):
        print(x)
    var = Variable(symbol='a', dimensions=[2, 2])
    print(var)

    x = np.array([['a', 'b'],
              ['c', 'd']], dtype=str)
    y = np.array([['e', 'f'],
                  ['g', 'h']], dtype=str)
    print(utils.np_matmul_str_matrices_2d(x, y))