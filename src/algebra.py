import re
from vars import globals


class Scalar:
    def __init__(self, name='', value=0.0, parity=1):
        self.value = value
        self.parity = parity
        self.name = name

    def __str__(self):
        return f'{self.name}'


class DualScalar(Scalar):
    def __init__(self, name='', parity=1):  # Parity may not be used
        Scalar.__init__(self, name, parity)
        assert name.endswith(globals.POS_SUFFIX) or name.endswith(globals.NEG_SUFFIX)


class Term:
    def __init__(self, scalars=[], parity=1):
        self.scalars = scalars
        self.parity = parity

    def __str__(self):
        sign = globals.EMPTY if self.parity == 1 else globals.MINUS
        return sign + globals.VARSEP.join([str(sc) for sc in
                                            self.scalars])


class Expression:
    def __init__(self, terms=[], parity=1):
        self.terms = []
        self.parity = parity

    def __str__(self):
        sign = globals.EMPTY if self.parity == 1 else globals.MINUS
        return sign + globals.TERMSEP.join([str(term) for term in self.terms])


if __name__ == '__main__':
    s1 = Scalar(name='a1p')
    s2 = Scalar(name='b2')
    t = Term(scalars=[s1, s2])
    print(t)
    ds = DualScalar(name='x1m')
    print(ds)