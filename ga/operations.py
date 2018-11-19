import pysdd
from gen.gen import generator
from model.model import Model
import random
import numpy as np
import time
from functools import reduce
from logic.cons import cons

class cross_over:
    def cross(self, model1, model2) -> Model:
        pass

class simple_subset_cross(cross_over):
    def cross(self, left, right):
        
        # Create an empty child
        new_left = empty_child(left)
        new_right = empty_child(right)
        
        # Get full sets
        left_rules = left.get_features()
        right_rules = right.get_features()
        
        # Get 2 subsets for each SDD
        left_A, left_B = split_collection(left_rules, [1,1])
        right_A, right_B = split_collection(right_rules, [1,1])
        
        # Create the new SDD's (models) from the subsets
        new_left = add_and_join(left, right, left_A, right_A, new_left)
        new_right = add_and_join(left, right, left_B, right_B, new_right)
        
        # Return the new models
        return new_left, new_right
        
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
        
class remove_mutation(mutation):
    def __init__(self):
        pass
        
    def mutate(self, model):
        
        features = model.get_features()
        random_rem = select_random_element(features)
        
        model.remove_factor(random_rem)
        
        return model
        
def select_random_element(elements):
    return random.choice(elements)
    
def empty_child(source):
    # TODO: transfer weights???
    domain_size = source.domain_size
    imgr = source.indicator_manager
    mgr = source.mgr
    return Model(domain_size, imgr, mgr)
    
def split_collection(collection, distr):
    
    n_sets = len(distr)
    n_coll = len(collection)
    distr = [i/sum(distr) for i in distr]
    sets = [[] for i in range(n_sets)]
    
    choices = np.random.choice(range(n_sets), n_coll, distr)
    
    for (c, f) in zip(choices, collection):
        sets[c].append(f)
        
    print([len(s) for s in sets])
    return sets
    
    
    
def add_and_join(left, right, left_subset, right_subset, target):
    left_sdd = sdd_from_subset(left, left_subset)
    right_sdd = sdd_from_subset(right, right_subset)
    union = left_subset + right_subset
    
    start = time.time()
    conjoined = left_sdd & right_sdd 
    end = time.time()
    
    target.from_pre_compiled_sdd(conjoined, union)
    
    return target
    
def sdd_from_subset(source, subset):
    # TODO: migrate this stuff to single place
    # TODO: Consider removing factors instead of adding factors
    enc_sdds = [e.get_sdd() for (_,_,e,_) in subset]
    default_sdd = cons(True).compile(source.mgr)
    conjunction = reduce( lambda x, y : x & y, enc_sdds, default_sdd)
    return conjunction
    
    
    
    
    
    
    
    
    
    
