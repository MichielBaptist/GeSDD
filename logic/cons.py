from logic.factor import factor
import pysdd

class cons(factor):
    def __init__(self, val):
        self.boolean_val = val
        
    def to_string(self):
        return str(self.boolean_val)
    
    def to_sdd(self, mgr):
        if self.boolean_val:
            return mgr.true()
        else:
            return mgr.false()
    
    def evaluate(self, world):
        return self.boolean_val
