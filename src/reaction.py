def suffix2index(suffix):
    ret = suffix.replace('p', '1')
    ret = ret.replace('m', '2')
    ret = [int(c) for c in ret]
    return ret


class Species:
    def __init__(self, name='', initconc=0.0):
        self.name = name
        self.init = initconc

    def __str__(self):
        return self.name


class Reaction:
    def __init__(self, reactants=[Species()], products=[Species()],
                 rate_constant='1.0', reversible=False):
        self.rs = reactants
        # also handles the case where there are no products
        self.ps = products if products is not [] else ['0']
        self.k = rate_constant
        self.reversible = reversible

    def __str__(self):
        return f"{self.k}, {'+'.join([r.name for r in self.rs])} -->" \
               f" {'+'.join([p.name for p in self.ps])}"

    def assign_concentrations(self):
        concentrations = []
        for r in self.rs:
            concentrations.append(f''':{r.name} => vars['{r.name}']\
                {str(suffix2index(r.name[1:]))}''')
        for p in self.ps:
            concentrations.append(f''':{p.name} => vars['{p.name}']\
                {str(suffix2index(p.name[1:]))}''')
        return concentrations


if __name__ == '__main__':
    crn = Reaction([Species(name='A11p')], [Species(name='B12m')], 'k')
    cs = crn.assign_concentrations()
    print(cs)
    print(crn)