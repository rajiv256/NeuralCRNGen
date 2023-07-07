class CRN:
    def __init__(self, reactants=[], products=[], rate_constant=[],
                 reversible=False):
        self.rs = reactants
        # also handles the case where there are no products
        self.ps = products if products is not [] else ['0']
        self.k = rate_constant
        self.reversible = reversible

    def __str__(self):
        return f"""{self.k}, {'+'.join(self.rs)} --> {'+'.join(self.ps)}"""


if __name__ == '__main__':
    crn = CRN(['a', 'b'], ['c', 'd'], 'k')
    print(crn)