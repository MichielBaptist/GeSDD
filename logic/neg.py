from logic.factor import factor
from logic.conj import conj
import pysdd
import numpy as np

class neg(factor):
    def __init__(self, negated_factor):
        self.id = None
        self.negated_factor = negated_factor

    def to_string(self):
        return "-(" + str(self.negated_factor) + ")"

    def to_sdd(self, mgr):

         #print(f"--neg-- {self.to_string()}")

        negated_factor = self.negated_factor.to_sdd(mgr)
        negation = mgr.negate(negated_factor)
        negation.ref()
        self.deref_all([negated_factor])

        #print(f"Neg: {negated_factor.ref_count()}")

        return negation

    def evaluate(self, world):
        return self.negated_factor.evaluate(world) == False

    def is_literal(self):
        return self.negated_factor.is_literal()

    def val(self):
        if not self.is_literal():
            print("Cannot find value of a negation of non-literal")
            return None
        return -1 * self.negated_factor.val()

    def all_conjoined_literals(self):
        if not isinstance(self.negated_factor, conj):
            print("Cannot find conjoined literals in negation if its negated factor is not conjunction")
            return None
        return self.negated_factor.all_conjoined_literals()

    def __eq__(self, other):
        if not isinstance(other, neg):
            return False

        return self.negated_factor == other.negated_factor
