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
        """Converts the Matrix2D object into an 2D nparray"""
        if self.data is not None:
            mat = np.array(self.data)
            if mat.ndim != 2:
                assert len(self.dims) == 2
                mat = mat.reshape(self.dims)
            return mat
        # If the matrix is a column vector, then give it just the indices and
        # no cartesian product is required.
        if self.dims[-1] == 1:
            indices = [str(i) for i in range(self.dims[0])]
        else:
            indices = [list(range(d)) for d in self.dims]
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
        ret = ''
        mat = self.matrix()
        rows, cols = self.dims 
        for r in range(rows):
            for c in range(cols):
                ret += str(mat[r][c]) + ' ' # Scalar object
            ret += '\n'
        return ret

    def __add__(self, mat):
        assert self.dims == mat.dims
        return Matrix2D(
            symbol=f'{self.symbol}PLUS{mat.symbol}',
            dims=self.dims,
            data=self.matrix() + mat.matrix()
        )

    def __mul__(self, mat):
        assert self.dims[1] == mat.dims[0]
        data = np.matmul(self.matrix(), mat.matrix())
        return Matrix2D(
            symbol=f'{self.symbol}{mat.symbol}',
            dims=[self.dims[0], mat.dims[1]],
            data=data
        )

    def _broadcast(self, mat):
        if self.dims == mat.dims:
            return mat
        newdata = np.broadcast_to(self.matrix(), mat.dims)
        assert (mat.dims[0]%self.dims[0]==0) and (mat.dims[1]%self.dims[1]==0)
        return Matrix2D(
            symbol=self.symbol + 'B',
            dims=mat.dims,
            data=newdata,
        )

    def _hadamard(self, mat):
        newdata = np.multiply(self.matrix(), mat.matrix())
        return Matrix2D(
            symbol=f'{self.symbol}HDMD{mat.symbol}',
            dims=list(newdata.shape),
            data=newdata
        )

    def _reshape(self, dims=[]):
        matflat = self.matrix().flatten()
        return Matrix2D(symbol=self.symbol, dims=dims, data=matflat)


    
    
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

    def __str__(self):
        ret = ''
        ret += f'lhs: {self.lhs.symbol}: \n{str(self.lhs)}\n'
        for r in self.rhs:
            ret += f'rhs: {r.symbol}: \n{str(r)}\n'
        return ret

    def dual_rail_crn(self):
        rhsprod = self.rhs[0]
        for i in range(1, len(self.rhs)):
            rhsprod = rhsprod*self.rhs[i]
        
        lhs_mat = self.lhs.matrix()
        rhs_mat = rhsprod.matrix()
        rhs_mat=np.broadcast_to(rhs_mat, lhs_mat.shape)

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
    print(x)
