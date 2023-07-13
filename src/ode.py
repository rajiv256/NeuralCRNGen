import itertools

import numpy as np

import utils
from src.reaction import Reaction, Species
from vars import globals


def transpose_variable(x):
    xT = Variable(
        symbol=x.symbol + "T", dimensions=reversed(x.dims),
        data=x.vectorize().transpose().tolist()
        )
    return xT


def create_dfdtheta(D, prefix='x'):
    """Creates a dfdtheta style array. Very niche function.
    """
    xvars = [Species(name=prefix + str(i)) for i in range(1, D + 1)]  # x1 x2
    # x3 ... xD
    zero = '0'
    xs = np.core.defchararray.add(zero, np.zeros((D, D ** 2), dtype=str))
    for i in range(xs.shape[0]):
        xs[i, D * i:D * i + D] = xvars
    return xs.tolist()


class MonomialODE:
    def __init__(self, lhs=Scalar(), rhs=[Scalar()], parity=1):
        self.lhs = lhs
        self.rhs = rhs
        self.parity = parity

    def __str__(self):
        return f"""d{self.lhs.name}/dt = 
        {''.join([r.name for r in self.rhs])}"""

    def catalytic_crn(self):
        """
        :rtype: Reaction
        """
        crn = Reaction(
            reactants=[utils.convert_dual_scalar_to_species(sc) for sc in
                       self.rhs],
            products=[utils.convert_dual_scalar_to_species(sc) for sc in
                      [self.lhs] + self.rhs], rate_constant=1.0,
            reversible=False
        )
        return crn

    def dual_rail(self):
        """
        :return: A [] of two multinomials
        """
        dr = utils.convert_expr_to_dual_rail(self.rhs, self.parity)
        # An example dual_rail variable looks as follows:
        # {'pos': ['ap|bp', 'an|bn'], 'neg': ['ap|bn', 'an|bp']}
        pos = MultinomialODE(
            monomials=[MonomialODE(
                lhs=self.lhs + globals.POS_SUFFIX, rhs=rs.split('|')
                ) for rs in dr['pos']]
        )
        neg = MultinomialODE(
            monomials=[MonomialODE(
                lhs=self.lhs + globals.NEG_SUFFIX, rhs=rs.split('|')
                ) for rs in dr['neg']]
        )
        return [pos, neg]


class MultinomialODE:
    def __init__(self, monomials=[]):
        assert len(monomials) > 0
        assert sum([m.lhs.name != monomials[0].lhs.name for m in monomials]) == 0
        self.lhs = monomials[0]
        self.multi_rhs = [m.rhs for m in monomials]
        self.monomials = monomials

    def __str__(self):
        return f"""d({self.lhs.name})/dt =
         {'+'.join([''.join(rhs.name) for rhs in self.multi_rhs])}"""

    def catalytic_crn(self):
        crn = [m.catalytic_crn() for m in self.monomials]
        return crn

    def dual_rail(self):
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

    def __init__(self, symbol='', dimensions=[], data=None):
        self.symbol = symbol
        self.dims = dimensions
        self.data = data

    def __str__(self):
        print(self.symbol)

    def vectorize(self):
        """
        :return: A np array of dtype str.
        """
        if self.data is not None:
            vect = np.array(self.data, dtype=str)
            return vect

        print("-->", self.dims)
        dim_indices = [list(range(1, d + 1)) for d in self.dims]
        concatenated_dims = itertools.product(*dim_indices)
        final_dims = np.array(
            [utils.tuple_to_string(t) for t in concatenated_dims], dtype=str
        )
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


class Matrix2D:
    def __init__(self, data=[['']]):
        self.data = data

    def __str__(self):
        for row in self.data:
            print(globals.SPACE.join(row))

    def _to_numpy_array(self):
        return np.array(self.data, dtype=str)


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
            print(rhs_vec, rhs_vec_list[i])
            rhs_vec = utils.np_matmul_str_matrices_2d(rhs_vec, rhs_vec_list[i])
        multinomials = []
        for i in range(lhs_vec.shape[0]):
            for j in range(lhs_vec.shape[1]):  # Assumption that it is all 2D
                # Added support for trailing + issue in np_matmul_str_matrices_2d
                multinomials.append(
                    MultinomialODE(
                        [MonomialODE(
                            lhs=lhs_vec[i, j], rhs=splt_item.split('|'),
                            parity=self.parity
                            ) for splt_item in rhs_vec[i, j].split('+') if
                            splt_item is not '']
                    )
                )

        # Remember that each monomial returns a multinomial in dual-rail
        dual_rail_multinomials = []
        for mn in multinomials:
            dual_rail_multinomials += mn.dual_rail()

        crn = []
        for mn in dual_rail_multinomials:
            crn += mn.catalytic_crn()

        return crn


if __name__ == '__main__':
    print("forward z")
    ode = ODESystem(
        lhs=Variable(symbol='z', dimensions=[2]),
        rhs=[Variable(symbol='P', dimensions=[2, 2]),
             Variable(symbol='z', dimensions=[2])], parity=-1
        )
    crn = ode.dual_rail_crn()
    for c in crn:
        print(c)