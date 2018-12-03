from model.model import Model
import math

class FitnessFunction:
    def of(self, model, data):
        pass
 
class SimpleFitness(FitnessFunction):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def of(self, model, data):
        ll = model.LL(data)
        size = model.sdd_size() 
        #print(ll, size)
        return  ll* self.alpha - math.log(size + 1)* self.beta
        
class fitness2(FitnessFunction):
    def __init__(self, alpha=0.5, base=0):
        self.alpha = alpha
        self.base = base
        
    def of(self, model, data):
        ll = model.LL(data)
        size = model.sdd_size()
       
        
        return max((ll - self.base) / (math.log(self.alpha * size + 1) + 1), 1e-2) # No negative fitness
