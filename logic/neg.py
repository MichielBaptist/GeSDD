from logic.factor import factor
import pysdd
import numpy as np

class neg(factor):
    def __init__(self, negated_factor):
        self.negated_factor = negated_factor
        self.cached_sdd = None
        super().__init__()

    def to_string(self):
        return "-(" + str(self.negated_factor) + ")"

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

        self.cached_sdd = mgr.negate(self.negated_factor.to_sdd(mgr))

        return self.cached_sdd


    def evaluate(self, world):
        return self.negated_factor.evaluate(world) == False

    def is_literal(self):
        return False

    def __eq__(self, other):
        if not isinstance(other, neg):
            return False

        return self.negated_factor == other.negated_factor
