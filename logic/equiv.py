from logic.factor import factor
from functools import reduce
from logic.neg import neg
from logic.conj import conj
from logic.disj import disj
import pysdd

class equiv(factor):
    def __init__(self, list_of_factors):
        self.list_of_factors = list_of_factors

    def to_string(self):
        return "(" + " <=> ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def to_sdd(self,mgr):


        # No need to deref conjs, happens automatically in disj
        left = conj(self.list_of_factors)
        right = conj([neg(f) for f in self.list_of_factors])
        equiv = disj([left, right]) # refs disjunction only

        
        return equiv.to_sdd(mgr)

    def evaluate(self, world):
        value = None
        for factor in self.list_of_factors:
            if value == None:
                value = factor.evaluate(world)

            factor_value = factor.evaluate(world)

            if factor_value != value:
                return False #One detected that is not the same value!

        return True

    def __eq__(self, other):
        if not isinstance(other, equiv):
            return False

        return self.list_of_factors == other.list_of_factors

    pass
