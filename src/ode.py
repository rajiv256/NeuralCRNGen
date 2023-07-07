from typing import List, Any

from utils import convert_var_to_chem
from crn import CRN


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
            reactants=[convert_var_to_chem(x) for x in self.rhs],
            products=[convert_var_to_chem(x) for x in [self.lhs] + self.rhs],
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

    def _enumerate(self):
        dim_indices = [range(1, d+1) for d in self.dims]
        concatenated_dims =

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