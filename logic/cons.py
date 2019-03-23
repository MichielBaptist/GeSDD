from logic.factor import factor
import pysdd

class cons(factor):
    def __init__(self, val):
        self.boolean_val = val
        self.cached_sdd = None
        super().__init__()

    def to_string(self):
        return str(self.boolean_val)

    def to_sdd(self, mgr):

        if self.boolean_val:
            sdd = mgr.true()
        else:
            sdd = mgr.false()

        return sdd

    def evaluate(self, world):
        return self.boolean_val

    def __eq__(self, other):
        if not isinstance(other, cons):
            return False

        return self.boolean_val == other.boolean_val
