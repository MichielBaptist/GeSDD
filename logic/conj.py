from logic.factor import factor
from functools import reduce
import pysdd
import numpy as np

class conj(factor):

    # List should be integers
    def __init__(self, list_of_factors):
        self.list_of_factors = list_of_factors
        self.cached_sdd = None
        super().__init__()

    def all_conjoined_literals(self):
        # This method returns a list of all literals of this conjunction
        # This works recursively ((a & b) & (c & d))=> [a, b, c, d]

        # 1) literals of this conjunction.
        literals_of_this = list(filter(lambda x: x.is_literal(), self.list_of_factors))
        literals_of_this = list(map(lambda x: x.val(), literals_of_this))

        # 2) find conjunctions inside this conjunction
        factors = list(filter(lambda x: not x.is_literal(), self.list_of_factors))
        conjunctions = list(filter(lambda x: isinstance(x, conj), factors))

        # 3) findliterals of these
        literals_of_others = list(map(lambda c: c.all_conjoined_literals(), conjunctions))
        literals_of_others = [l for sub in literals_of_others for l in sub ]

        # 4) Join both
        uniques = literals_of_this + literals_of_others
        uniques = np.unique(uniques)

        # 5) done
        return uniques

    def to_string(self):
        return "(" + " & ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def ref(self):

        if self.cached_sdd == None:
            return

        if self.cached_sdd.garbage_collected():
            self.cached_sdd = None
            return

        self.cached_sdd.ref()

    def deref(self):
        if self.cached_sdd == None:
            return

        if self.cached_sdd.garbage_collected(): #Already derefd
            self.cached_sdd = None
            return

        self.cached_sdd.deref()

    def to_sdd(self, mgr):

        if self.cached_sdd != None and not self.cached_sdd.garbage_collected():
            return self.cached_sdd

        #important to call "compile" and not call "to_sdd"
        sdd_of_factors = map(lambda x: x.compile(mgr), self.list_of_factors)
        conjunction_of_sdd = reduce( lambda x,y: x & y, sdd_of_factors )

        self.cached_sdd = conjunction_of_sdd

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
