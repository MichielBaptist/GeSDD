from logic.factor import factor
import pysdd

class impl(factor):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.cached_sdd = None
        super().__init__()

    def to_string(self):
        return self.left.to_string() + " => " + self.right.to_string()

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

        self. cached_sdd = mgr.negate(self.left.to_sdd(mgr)) | self.right.to_sdd(mgr)

        return self.cached_sdd

    def evaluate(self, world):
        return not self.left.evaluate(world) or self.right.evaluate(world)

    def __eq__(self, other):
        if not isinstance(other, impl):
            return False

        return self.left == other.left and self.right == other.right
