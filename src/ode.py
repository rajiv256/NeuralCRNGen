import itertools
import utils
from crn import CRN
import numpy as np


class MonomialODE:
    def __init__(self, lhs='', rhs=['a', 'b'], parity=1):
        self.lhs = lhs
        self.rhs = rhs
        self.parity = parity

    def __str__(self):
        return f"""d{self.lhs}/dt = {''.join(self.rhs)} """

    def catalytic_crn(self):
        """
        :rtype: CRN
        """
        crn = CRN(reactants=[utils.convert_var_to_chem(x) for x in self.rhs],
                  products=[utils.convert_var_to_chem(x) for x in
                            [self.lhs] + self.rhs], rate_constant='1.0',
                  reversible=False)
        return crn

    def dual_rail(self):
        """
        :return: A [] of two multinomials
        """
        dr = utils.convert_expr_to_dual_rail(self.rhs, self.parity)
        # An example dual_rail variable looks as follows:
        # {'pos': ['ap|bp', 'an|bn'], 'neg': ['ap|bn', 'an|bp']}
        pos = MultinomialODE(
            monomials=[MonomialODE(lhs=self.lhs + 'p', rhs=rs.split('|')) for rs
                       in dr['pos']])
        neg = MultinomialODE(
            monomials=[MonomialODE(lhs=self.lhs + 'n', rhs=rs.split('|')) for rs
                       in dr['neg']])
        return [pos, neg]


class MultinomialODE:
    def __init__(self, monomials=[]):
        assert len(monomials) > 0
        assert sum([m.lhs != monomials[0].lhs for m in monomials]) == 0
        self.lhs = monomials[0].lhs
        self.multi_rhs = [m.rhs for m in monomials]
        self.monomials = monomials

    def __str__(self):
        return f"""d({self.lhs})/dt =
         {'+'.join([''.join(rhs) for rhs in self.multi_rhs])}"""

    def catalytic_crn(self):
        crn = [m.catalytic_crn() for m in self.monomials]
        return crn

    def dual_railss(self):
        """
        :return:A [] of multinomials
        """
        ret = []
        for m in self.monomials:
            ret += m.dual_rail()
        return ret


class Variable:
    """
    Vector variable
    """

    def __init__(self, symbol='', dimensions=[]):
        self.symbol = symbol
        self.dims = dimensions

    def __str__(self):
        print(self.symbol)

    def vectorize(self):
        """
        :return: A np array of dtype str.
        """
        print("-->", self.dims)
        dim_indices = [list(range(1, d + 1)) for d in self.dims]
        concatenated_dims = itertools.product(*dim_indices)
        final_dims = np.array(
            [utils.tuple_to_string(t) for t in concatenated_dims], dtype=str)
        # If it is a list, make it a column vector, otherwise keep it the same.
        if len(self.dims) == 1:
            reshape_dims = self.dims + [1]
        else:
            reshape_dims = self.dims
        final_dims = final_dims.reshape(reshape_dims)
        vect = np.core.defchararray.add(self.symbol, final_dims)
        return vect

    def __str__(self):
        var = self.vectorize()
        return str(var)


class ODESystem:
    """
    For ODEs with RHSs that can be represented as vector-jacobians.
    """

    def __init__(self, lhs=Variable(), rhs=[Variable()], parity=1):
        """Takes in a set of multinomials and saves them in a compressed
        matrix form."""
        self.lhs = lhs
        self.rhs = rhs
        self.parity = parity

    def dual_rail_crn(self):
        lhs_vec = self.lhs.vectorize()

        rhs_vec_list = [r.vectorize() for r in self.rhs]
        rhs_vec = rhs_vec_list[0]
        for i in range(1, len(rhs_vec_list)):
            rhs_vec = utils.np_matmul_str_matrices_2d(rhs_vec, rhs_vec_list[i])

        multinomials = []
        for i in range(lhs_vec.shape[0]):
            for j in range(lhs_vec.shape[1]):  # Assumption that it is all 2D
                multinomials.append(MultinomialODE(
                    [MonomialODE(lhs=lhs_vec[i, j], rhs=splt_item.split('|'),
                                 parity=self.parity)
                     for splt_item in rhs_vec[i, j].split('+')]))

        # Remember that each monomial returns a multinomial in dual-rail
        dual_rail_multinomials = []
        for mn in multinomials:
            dual_rail_multinomials += mn.dual_railss()

        crn = []
        for mn in dual_rail_multinomials:
            crn += mn.catalytic_crn()

        return crn


if __name__ == '__main__':
    print("forward z")
    ode = ODESystem(lhs=Variable(symbol='z', dimensions=[2]),
                    rhs=[Variable(symbol='P', dimensions=[2, 2]),
                         Variable(symbol='z', dimensions=[2])], parity=-1)
    crn = ode.dual_rail_crn()
    for c in crn:
        print(c)