from logic.factor import factor
import pysdd
import numpy as np

class lit(factor):
    def __init__(self, literal_int):
        self.literal_int = literal_int
    
    def to_string(self):
        return str(self.literal_int)
    
    def to_sdd(self, mgr):
        
        sdd = mgr[int(self.literal_int)]
        
        return sdd
    
    def val(self):
        return self.literal_int
        
    def evaluate(self, world):
        index = abs(self.literal_int) - 1 # Index in world
        sign = np.sign(self.literal_int)     # Sign of this literal
        
        if world[index] == True and sign > 0:
            return True
        elif world[index] == False and sign < 0:
            return True
        else:
            return False
        
    def is_literal(self):
        return True
