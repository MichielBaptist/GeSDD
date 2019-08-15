from logic.factor import factor
from functools import reduce
import pysdd

class disj(factor):

    # List should be integers
    def __init__(self, list_of_factors):
        self.list_of_factors = list_of_factors

    def to_string(self):
        return "(" + " | ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def to_sdd(self, mgr):

        #print(f"--disj-- {self.to_string()}")

        sdd_of_factors = [x.to_sdd(mgr) for x in self.list_of_factors]
        disjunction = reduce( lambda x,y: x | y, sdd_of_factors )

        disjunction.ref()
        self.deref_all(sdd_of_factors)

        #for sdd in sdd_of_factors:
        #    sdd.deref()


        #print(f"Disj: {[sdd.ref_count() for sdd in sdd_of_factors]}")

        return disjunction

    def evaluate(self, world):
        for factor in self.list_of_factors:
            if factor.evaluate(world) == True:
                return True
        return False

    def __eq__(self, other):
        if not isinstance(other, disj):
            return False

        return self.list_of_factors == other.list_of_factors

    pass
