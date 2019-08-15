from model.model import Model
import math
from pysdd.sdd import SddManager

import utils.string_utils as stru

class FitnessFunction:
    def of(self, model, data, set_id):
        pass

    def of_wrap(self, pop, data, set_id):
        return [self.of(ind, data, set_id) for ind in pop]

class globalFit(FitnessFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def of_wrap(self, pop, data):
        sz = [mdl.sdd_size() for mdl in pop]
        lls= [mdl.LL(data) for mdl in pop]

        fits = [ self.f(i, sz, lls) for i, mdl in enumerate(pop)]

        def srt(tpl):
            return tpl[2]
        tbl = [("Size", "LL(T)", "Fitness")]
        tbl = tbl + sorted( list(zip(sz, lls, fits)), key=srt)
        print(stru.pretty_print_table(tbl))


        return fits

    def f(self, i, sz, lls):
        return lls[i] - min(lls) + self.alpha*math.log(max(sz) - sz[i] + 1)

class fitness3(FitnessFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def of(self, model, data):
        ll = model.LL(data)
        size = model.sdd_size()
        return ll - self.alpha * size

    def __str__(self):
        lines = [
        "Fitness 3",
        "ll - a * |sdd|",
        f"-->alpha: {self.alpha}"
        ]
        return "\n".join(lines)


class SimpleFitness(FitnessFunction):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def of(self, model, data):
        ll = model.LL(data)
        size = model.sdd_size()
        #print(ll, size)
        return  ll* self.alpha - math.log(size + 1)* self.beta

    def __str__(self):
        lines = [
            "Fitness 1",
            "a*ll - b*Log(|sdd|)",
            f"--> alpha: {self.alpha}",
            f"--> beta:  {self.beta}"
        ]
        return "\n".join(lines)

class fitness2(FitnessFunction):
    def __init__(self, alpha=0.5, base=0):
        self.alpha = alpha
        self.base = base

    def of(self, model, data):
        ll = model.LL(data)
        size = model.sdd_size()


        return max((ll - self.base) / (math.log(self.alpha * size + 1) + 1), 1e-5) # No negative fitness

    def __str__(self):
        lines = [
            "Fitness 2",
            "LL gain per log(n)",
            "--> F(i) = ll - base / (log(alpha*n + 1) + 1)   If F(i) >= 0",
            "         = 0                                    Else ",
            f"--> alpha: {self.alpha}"
        ]
        return "\n".join(lines)

class fitness4(FitnessFunction):
    def __init__(self, alpha=0.5, base=0):
        self.alpha = alpha
        self.base = base

    def of(self, model, data):
        ll = model.LL(data)
        size = model.sdd_size()


        return max((ll - self.base) - self.alpha * (math.log(size+1)), 1e-5) # No negative fitness

    def __str__(self):
        lines = [
            "Fitness 2",
            "LL gain per log(n)",
            "--> F(i) = ll - base - (alpha*log(n + 1))   If F(i) >= 0",
            "         = 0                                    Else ",
            f"--> alpha: {self.alpha}"
        ]
        return "\n".join(lines)

class fitness5(FitnessFunction):
    def __init__(self, alpha, base):
        self.alpha = alpha
        self.base = base

    def of(self, model, data, set_id):
        ll = model.LL(data, set_id)
        size = model.sdd_size()
        return max((ll - self.base) - self.alpha * size, 1e-6)

    def __str__(self):
        lines = [
        "Fitness 5",
        "(ll-base) - a * |sdd|",
        f"-->alpha: {self.alpha}"
        ]
        return "\n".join(lines)
