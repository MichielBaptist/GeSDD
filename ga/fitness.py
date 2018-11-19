from model.model import Model

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
        print(ll, size)
        return  ll* self.alpha - size* self.beta
