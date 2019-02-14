import time
import numpy as np

def get_timer():
    return timer()
    
class timer:
    def __init__(self):
        self.time = None
        self.times = {}
        
    def start(self):
        self.now()
        
    def now(self):
        self.time = time.time()
        
    def t(self, name):
        if not name in self.times:
            self.times[name] = [self.cut()]          
        else:
            self.times[name].append(self.cut())
        
    def cut(self):
        diff = time.time() - self.time
        self.now()
        return diff
        
    def avg(self):
        return [(k, np.mean(v)) for (k, v) in self.times.items()]
        
    def sum(self):
        return [(k, sum(v)) for (k, v) in self.times.items()]
