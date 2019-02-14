from logic.factor import factor
from functools import reduce
import pysdd

class equiv(factor):
    def __init__(self, list_of_factors):
        self.list_of_factors = list_of_factors
        self.dirty = True
        self.sdd = None
        super().__init__()

    def to_string(self):
        return "(" + " <=> ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def to_sdd(self,mgr):

        sdd_of_factors = list(map(lambda x: x.to_sdd(mgr), self.list_of_factors)) # Get the SDD of each factor
        conjunction_of_factors = reduce( lambda x,y: x & y, sdd_of_factors )      # Conjoin all

        sdd_of_factors_negated = map(lambda x:mgr.negate(x), sdd_of_factors)      # Get the negated SDD of each
        conjunction_of_factors_negated = reduce( lambda x,y: x & y, sdd_of_factors_negated ) # Conjoin all

        # A <=> B <=> ... <=> Z == (A ^ B ^ ... ^ Z) | (-A ^ -B ^ ... ^ -Z)
        equiv_sdd = mgr.disjoin(conjunction_of_factors, conjunction_of_factors_negated)

        return equiv_sdd

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
