from logic.factor import factor
from functools import reduce
import pysdd
import numpy as np

class conj(factor):

    # List should be integers
    def __init__(self, list_of_factors):
        self.id = None
        self.list_of_factors = self.flatten_conjuntions(list_of_factors)

    def flatten_conjuntions(self, lst):
        return self.flatten([self.flatten_conjunctor(c) for c in lst])

    def flatten(self, lst):

        def flatten_el(e):
            if isinstance(e, list):
                return self.flatten(e)
            else:
                return [e]

        flat_lol = [flatten_el(e) for e in lst]
        #print(f"flat: {flat_lol}")
        #print(f"returning: {[e for lst in flat_lol for e in lst]}")
        return [e for lst in flat_lol for e in lst]


    def flatten_conjunctor(self, c):
        if isinstance(c, conj):
            return self.flatten_conjuntions(c.list_of_factors)
        else:
            return c

    def all_conjoined_literals(self):
        # This method returns a list of all literals of this conjunction
        # This works recursively ((a & b) & (c & d))=> [a, b, c, d]

        # 1) literals of this conjunction. (also include negations)
        literals_of_this = list(filter(lambda x: x.is_literal(), self.list_of_factors))
        literals_of_this = list(map(lambda x: x.val(), literals_of_this))

        # 2) find conjunctions inside this conjunction
        factors = list(filter(lambda x: not x.is_literal(), self.list_of_factors))

        # 3) findliterals of these
        literals_of_others = list(map(lambda c: c.all_conjoined_literals(), factors))
        literals_of_others = [l for sub in literals_of_others for l in sub ]

        # 4) Join both
        uniques = literals_of_this + literals_of_others
        uniques = list(np.unique(uniques))

        # 5) done
        return uniques

    def to_string(self):
        return "(" + " & ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def to_sdd(self, mgr):

        #print(f"--conj {self.to_string()} -- ")

        sdd_of_factors = [fct.to_sdd(mgr) for fct in self.list_of_factors]
        conjunction = reduce( lambda x,y: x & y, sdd_of_factors )

        conjunction.ref()
        self.deref_all(sdd_of_factors)

        #print(f"Conj: {[sdd.ref_count() for sdd in sdd_of_factors]}")

        return conjunction

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
