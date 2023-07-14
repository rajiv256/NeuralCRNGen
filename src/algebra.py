import re
from vars import globals


class Expression:
    def __init__(self, terms=[]):
        self.terms = terms

    def __str__(self):
        if self.terms == []:
            return "Empty"
        return globals.TERMSEP.join([str(term) for term in self.terms])

    def _add(self, e):
        return Expression(
            terms=self.terms + e.terms
        )

    def dual_rail(self):
        ret = []
        for term in self.terms:
            ret.append(term.dual_rail())
        return ret

    def _scalars(self):
        terms = self.terms
        ret = []
        for term in terms:
            ret += term.scalars
        return ret


class DualRail:
    def __init__(self, pos=Expression(), neg=Expression()):
        self.pos = pos
        self.neg = neg

    def __str__(self):
        return f'''pos: {str(self.pos)}\nneg: {str(self.neg)}'''


class Term(Expression):
    def __init__(self, scalars=[]):
        self.scalars = scalars
        self.name = globals.EMPTY.join([sc.name for sc in self.scalars])
        super().__init__(terms=[self])

    def __str__(self):
        sign = ''
        return sign + globals.VARSEP.join(
            [str(sc) for sc in self.scalars]
            )

    def dual_rail(self):
        dr_scalars = [sc.dual_rail() for sc in self.scalars]
        e = dr_scalars[0]
        for i in range(1, len(dr_scalars)):
            e = mult_dual_rail(e, dr_scalars[i])
        return e


class Scalar(Term):
    def __init__(self, name='0', value=0.0):

        self.value = value
        self.name = name
        super().__init__(scalars=[self])

    def __str__(self):
        return f'{self.name}'

    def dual_rail(self):
        return DualRail(
            pos=Scalar(name=self.name + globals.POS_SUFFIX),
            neg=Scalar(name=self.name + globals.NEG_SUFFIX)
        )

    def to_species_str(self):
        return self.name.upper() + self.name[1:]


def mult_terms(t1: Term, t2: Term):
    return Term(
        scalars=t1.scalars + t2.scalars
    )


def mult_expressions(e1: Expression, e2: Expression):
    all_terms = []
    for t1 in e1.terms:
        for t2 in e2.terms:
            all_terms.append(mult_terms(t1, t2))
    return Expression(
        terms=all_terms
    )


def mult_dual_rail(dr_1: DualRail, dr_2: DualRail):
    return DualRail(
        pos=add_expressions(
            mult_expressions(dr_1.pos, dr_2.pos),
            mult_expressions(dr_1.neg, dr_2.neg)
        ),
        neg=add_expressions(
            mult_expressions(dr_1.pos, dr_2.neg),
            mult_expressions(dr_1.neg, dr_2.pos)
        )
    )


def add_expressions(e1: Expression, e2: Expression):
    if e1 is None:
        return e2
    if e2 is None:
        return e1
    return Expression(
        terms=e1.terms+e2.terms,
    )


if __name__ == '__main__':
    s1 = Scalar(name='a1')
    s2 = Scalar(name='b2')
    s3 = Scalar(name='c3')
    s4 = Scalar(name='d4')
    e1 = Expression(terms=[s1, s2])
    e2 = Expression(terms=[s3, s4])
    t = Term(scalars=[s1, s2])
    print(t.dual_rail())

    e = mult_expressions(e1, e2)
    # e = mult_expressions(e, s3)
    print(e, [str(t) for t in e.terms])