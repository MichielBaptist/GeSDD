import pysdd
from gen.gen import generator
from model.model import Model

class cross_over:
    def cross(self, model1, model2) -> Model:
        pass

class simple_subset_cross(cross_over):
    def cross(self, left, right):
        
        left_rules = left.get_f()
        right_rules = right.get_f()
        
class mutation:
    def mutate(self, model) -> Model:
        pass
        
class add_mutation(mutation):
    def __init__(self):
        pass
        
    def mutate(self, model):
        
        # Domain size
        domain_size = model.domain_size
        
        gen = generator(domain_size)
        (random_f, random_w) = gen.random_feature()
        
        model.add_factor(random_f, random_w)
        
        return model
        
        
