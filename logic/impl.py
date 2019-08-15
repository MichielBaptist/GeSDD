from logic.factor import factor
import pysdd

class impl(factor):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def to_string(self):
        return self.left.to_string() + " => " + self.right.to_string()

    def to_sdd(self, mgr):
        lsdd = mgr.negate(self.left.to_sdd(mgr))
        rsdd = self.right.to_sdd(mgr)
        implication = lsdd | rsdd
        implication.ref()
        lsdd.deref()
        rsdd.deref()

        return implication

    def evaluate(self, world):
        return not self.left.evaluate(world) or self.right.evaluate(world)

    def __eq__(self, other):
        if not isinstance(other, impl):
            return False

        return self.left == other.left and self.right == other.right
