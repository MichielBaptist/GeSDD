from logic.factor import factor
from functools import reduce
import pysdd

class conj(factor):

    # List should be integers
    def __init__(self, list_of_factors):
        self.list_of_factors = list_of_factors
        super().__init__()

    def to_string(self):
        return "(" + " & ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def to_sdd(self, mgr):
        sdd_of_factors = map(lambda x: x.to_sdd(mgr), self.list_of_factors)
        conjunction_of_sdd = reduce( lambda x,y: x & y, sdd_of_factors )

        return conjunction_of_sdd

    def evaluate(self, world):
        for factor in self.list_of_factors:
            if factor.evaluate(world) == False:
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, conj):
            return False

        return self.list_of_factors == other.list_of_factors

    pass
