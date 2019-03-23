from model.model import Model
import math

class FitnessFunction:
    def of(self, model, data):
        pass

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


        return max((ll - self.base) / (math.log(self.alpha * size + 1) + 1), 1e-2) # No negative fitness

    def __str__(self):
        lines = [
            "Fitness 2",
            "LL gain per log(n)",
            "--> F(i) = ll - base / (log(alpha*n + 1) + 1)   If F(i) >= 0",
            "         = 0                                    Else ",
            f"--> alpha: {self.alpha}"
        ]
        return "\n".join(lines)
