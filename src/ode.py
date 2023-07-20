import itertools

import numpy as np

import utils
from src.algebra import DualRail, Scalar, Term, Expression
from src.reaction import Reaction
from vars import globals


class Matrix2D:
    def __init__(self, symbol='', dims=[], data=None):
        """
        `data` is a listoflist of algebra.Scalars
        """
        if data is None:
            assert len(dims) == 2  # Only 2D matrices allowed
        self.symbol = symbol
        self.dims = dims
        # If `data` is given, the matrix conversion proceeds solely through that
        # data. Otherwise, we create matrix normally with indices.
        self.data = data

    def matrix(self):
        if self.data is not None:
            mat = np.array(self.data)
            if mat.ndim != 2:
                assert len(self.dims) == 2
                mat = mat.reshape(self.dims)
            return mat
        # If the matrix is a column vector, then give it just the indices and
        # no cartesian product is required.
        if self.dims[-1] == 1:
            indices = [str(i) for i in range(1, 1+self.dims[0])]
        else:
            indices = [list(range(1, d + 1)) for d in self.dims]
            indices = itertools.product(*indices)
            indices = [utils.tuple_to_string(t) for t in indices]
        scalars = np.array(
            [Scalar(name=self.symbol + index) for index in indices]
        )
        # If it is a 1D, convert it into 2D column matrix
        reshape_dims = self.dims
        mat = scalars.reshape(reshape_dims)
        self.data = mat.tolist()
        return mat

    def __str__(self):
        mat = self.matrix()
        return str(mat)


def transpose_matrix(mat):
    matT = Matrix2D(
        symbol=mat.symbol + "T", dims=list(reversed(mat.dims)),
        data=mat.matrix().transpose().tolist()
    )
    return matT


def create_dfdtheta(D, prefix='x'):
    xs = []

    for i in range(D):
        for j in range(D**2):
            if len(xs) <= i:
                xs.append([])
            if D*i <= j < (i+1)*D:
                xs[i].append(Scalar(name=prefix + str(j-D*i + 1)))
            else:
                xs[i].append(Expression())

    xs = np.array(xs)
    return xs


class MultinomialODE:
    def __init__(self, lhs=Scalar(), rhs=Expression()):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"""d{str(self.lhs)}/dt = {str(self.rhs)}"""


class MonomialODE(MultinomialODE):
    def __init__(self, lhs=Scalar(), rhs=Term(), parity=1):
        self.lhs = lhs
        self.rhs = rhs
        self.parity = parity
        super().__init__(
            lhs=lhs, rhs=Expression(terms=[self.rhs])
        )

    def __str__(self):
        return f"""d{str(self.lhs)}/dt = {str(self.rhs)}"""


def monomial_dual_rail(mono, parity=1):
    lhs_dr = mono.lhs.dual_rail()
    rhs_dr = mono.rhs.dual_rail()
    if parity == 1:
        ret = [MultinomialODE(lhs_dr.pos, rhs_dr[0].pos),
               MultinomialODE(lhs_dr.neg, rhs_dr[0].neg)]
    if parity == -1:
        ret = [MultinomialODE(lhs_dr.pos, rhs_dr[0].neg),
               MultinomialODE(lhs_dr.neg, rhs_dr[0].pos)]
    return ret


def multinomial_dual_rail(multi, parity=1):
    ret = []
    for term in multi.rhs.terms:
        ret += monomial_dual_rail(
            MonomialODE(lhs=multi.lhs, rhs=term), parity=parity
        )
    return ret


def monomial_catalytic_crn(mono):
    """Convert a monomial expression to a catalyic crn
    :returns --> Reaction type
    """
    crn = Reaction(
        reactants=[utils.convert_dual_scalar_to_species(sc) for sc in
                   mono.rhs._scalars()],
        products=[utils.convert_dual_scalar_to_species(sc) for sc in
                  [mono.lhs] + mono.rhs._scalars()], rate_constant=1.0,
        reversible=False
    )
    return crn


def multinomial_catalytic_crn(multi):
    crn = []
    for term in multi.rhs.terms:
        m = MonomialODE(multi.lhs, term)
        crn += [monomial_catalytic_crn(m)]
    return crn


class ODESystem:
    def __init__(self, lhs=Matrix2D(dims=[0, 0]), rhs=[Matrix2D(dims=[0,0])], parity=1):
        self.lhs = lhs
        self.rhs = rhs
        self.parity = parity

    def dual_rail_crn(self):
        lhs_mat = self.lhs.matrix()
        rhs_mat_list = [r.matrix() for r in self.rhs]
        rhs_mat = rhs_mat_list[0]

        for i in range(1, len(rhs_mat_list)):
            rhs_mat = utils.np_matmult_scalar_matrices_2d(
                rhs_mat, rhs_mat_list[i]
            )
        multinomials = []
        for i in range(lhs_mat.shape[0]):
            for j in range(lhs_mat.shape[1]):
                multinomials.append(
                    MultinomialODE(
                        lhs=lhs_mat[i, j], rhs=rhs_mat[i, j]
                    )
                )
        ret = []
        for item in multinomials:
            ret += multinomial_dual_rail(item, parity=self.parity)
        crn = []
        for mn in ret:
            crn += multinomial_catalytic_crn(mn)
        return crn


if __name__ == '__main__':
    ode = ODESystem(
        lhs=Matrix2D(symbol='z', dims=[2, 1]),
        rhs=[Matrix2D(symbol='p', dims=[2, 2]),
             Matrix2D(symbol='z', dims=[2, 1])],
        parity=-1
    )
    crn = ode.dual_rail_crn()
    for c in crn:
        print(c)