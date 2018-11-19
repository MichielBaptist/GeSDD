import numpy as np
import pysdd
import math
import random
import cython
import matplotlib.pyplot as plt
import sklearn

from time import time

from graphviz import Source
from pysdd.sdd import Vtree, SddManager, WmcManager, Fnf
from functools import reduce
from scipy.optimize import minimize

from model.model import Model

from logic.conj import conj
from logic.disj import disj
from logic.lit  import lit
from logic.cons import cons
from logic.factor import factor
from logic.equiv import equiv

class generator:
    
    def __init__(self, domain_size, indicator_manager, manager = None, fn_sparseness = 0.25, operation_distr = [20,4,1], min_feature_size = 2, feature_size_factor=1.5, subgroup_size_factor = 1.0, remove_after_select_p = 0.95, negation_prob = 0.5, max_nb_factors = 10):
        self.domain_size = domain_size
        self.feature_number_sparseness = fn_sparseness
        self.operation_distr = [x/sum(operation_distr) for x in operation_distr]
        self.min_feature_size = min_feature_size
        self.feature_size_factor = feature_size_factor
        self.subgroup_size_factor = subgroup_size_factor
        self.remove_after_select_p = remove_after_select_p
        self.negation_prob = negation_prob
        self.max_nb_factors = max_nb_factors
        self.indicator_manager = indicator_manager
        self.manager = manager
        pass
    
    def set_domain_size(self, domain_size):
        self.domain_size
    
    def gen_n(self, n):
        return [self.gen() for i in range(n)]
        
    # Returns a randomly generated model
    def gen(self):
        modl = Model(self.domain_size, max_nb_factors = self.max_nb_factors)
        modl.set_indicator_manager(self.indicator_manager)
        modl.set_manager(self.manager)
        
        # 1) Select the number of features 
        n_factors = self.random_feature_n()
        
        # 2) Create random features
        r_features = [self.random_feature() for i in range(n_factors)]
        
        # 3) Add each of the generated features to the model
        for (f,w) in r_features:
            modl.add_factor(f,w)
            
        return modl
    
    def random_feature_n(self):
        pct = np.random.exponential(self.feature_number_sparseness)
        n = int((pct * self.domain_size))
            
        return n + 1
    
    def random_feature(self):
        rand_f = self.random_feature_f()
        rand_w = self.random_feature_w()        
        return (rand_f, rand_w)
    
    def random_feature_f(self):
        rand_fs = self.random_feature_s()
        rand_subselection = self.random_subgroup(rand_fs, self.all_vars())
        rand_subselection = self.random_negation(rand_subselection)
        rand_subselection = [lit(x) for x in rand_subselection]
        
        rand_feature = self.random_composition_rec(rand_subselection)
        
        #print(rand_feature)
        return rand_feature
    
    def random_negation(self, group):
        return [self.random_negation_b(x) for x in group]
    
    def random_negation_b(self, x):
        if self.coin_flip(self.negation_prob):
            return -x
        else:
            return x
            
    def random_composition_base(self, group):
        operation = self.random_operation()
        form = operation(group)
        return form
    
    def random_operation(self):
        return np.random.choice([disj, conj, equiv], 1, p =self.operation_distr)[0]
        
    def random_composition_rec(self, group):
        if len(group) == 1: # Done recursion
            return group[0] # Return the head composition

        # 1 select and delete with certain prob. 0.8 chance of deletion.
        selected, remaining = self.select_and_delete(group)

        # 2 compose those selected randomly 
        composed_selection = self.random_composition_base(selected)
        
        # 3 Add the composed_selection as a factor back in the list and re-do process:
        remaining.append(composed_selection)
        
        return self.random_composition_rec(remaining)
    
    def select_and_delete(self, group):
        rand_subset_size = self.random_subgroup_size(len(group))                # Choose random subgroup size (triangular)
        rand_subset_ind = np.random.choice(len(group), rand_subset_size, False) # Choose the subset
        rand_subset_ind = sorted(rand_subset_ind, reverse=True)
        
        rand_subset = []
        for i in rand_subset_ind:
            rand_subset.append(group[i])
            if self.coin_flip(self.remove_after_select_p):
                del group[i]
        
        return rand_subset, group
        
    def random_subgroup_size(self, scale):
        sz = np.random.triangular(2, 2, scale+1)
        return int(sz)
    
    def random_subgroup(self, size, group):
        if size > len(group):
            size = len(group)
        return np.random.choice(group, size, False)
    
    def all_vars(self):
        return [x for x in range(1,self.domain_size + 1)]
    
    def random_feature_s(self):
        extra_size = np.random.normal(0, self.feature_size_factor)
        return int(abs(extra_size) + self.min_feature_size)
    
    def random_feature_w(self):
        return  abs(np.random.normal(0,1))
    
    def coin_flip(self, p):
        return np.random.uniform() < p
        

def random_data(var, worlds):
    return np.random.randint(2, size=(worlds,var))
    
