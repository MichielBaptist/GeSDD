import numpy as np
import pysdd
import math
import random

import matplotlib.pyplot as plt
import sklearn

from time import time

from graphviz import Source
from pysdd.sdd import Vtree, SddManager, WmcManager, Fnf
from functools import reduce
from scipy.optimize import minimize

from logic.conj import conj
from logic.disj import disj
from logic.lit  import lit
from logic.cons import cons
from logic.factor import factor
from logic.equiv import equiv

from model.indicator_manager import indicator_manager

class Model:
    """
    Class representing a logical Theory, just a conjunction of Rules. Each rule has a weight.
    """
    
    # --------------------- init stuff --------------------
    def __init__(self, domain_size, indicator_manager = None, manager = None, dynamic_update=False, max_nb_factors=10,
                    pool_range = None):
        # Method must get a manager! 
        self.factors = []              # List of (formula, weight, encoding, indicator) pairs. endocing contains SDD
        self.factor_stack = []         # List of (formula, weight) pairs to be added to the SDD
        self.domain_size = domain_size # How many literals can this theory have?
        self.nb_factors = 0
        self.max_nb_factors = max_nb_factors
        
        self.indicator_manager = indicator_manager
        self.mgr = manager
        self.sdd = None                # Compiled SDD of the encoding (not of conj of factors)
        
        self.dynamic_update = dynamic_update  # Should we update the SDD each time we add a factor?
        self.dirty = False                    # Is there need for recomputation?
        
        self.add_initial_literals()           # Add the initial factors which give weight to literals A, B, C ....
        
        # Nothing at start
        self.Z = None
        self.probs = None
        
        if self.indicator_manager == None:
            self. indicator_manager = self.init_indicator_manager(self.domain_size, self.max_nb_factors)
    
    def add_initial_literals(self):
        for i in range(self.domain_size):
            self.add_literal_factor(lit(i + 1))
    
    def init_indicator_manager(self, domain_size, max_number_factors):
        bottom = domain_size + 1
        top = bottom + max_number_factors
        return indicator_manager(range(bottom, top))
        
    def from_pre_compiled_sdd(self, sdd, factors):
        if self.sdd != None:
            print("Model has sdd already!")
        if self.nb_factors > 0:
            print("Model has factors already!")
            
        self.factors = factors
        self.nb_factors = len(factors)
        self.sdd = sdd
        
    # ------------------ Training ----------------
    
    def fit(self, worlds):
        # worlds: [[{1,0}]]
        
        # 1) Count the worlds for each factor in the SDD:
        self.factor_counts = self.count(self.get_factors(), worlds)
        self.number_worlds = len(worlds)
        
        self.wmc = self.sdd.wmc()
        
        # 2) define the objective
        objective = self.objective_function
        
        # 3) define the gradient of the objective
        gradient = self.objective_function_gradient
        
        # 4) Optimize
        res = minimize(objective, x0 = self.get_factor_weights(), jac = None, method="BFGS",
                      options={'gtol': 1e-1, 'disp': True})
        
        # 5) Set the weights correct!
        self.set_factor_weights(res.x)
        
        self.dirty = True        
        self.wmc = None
        
    def objective_function(self, factor_weights):
        # 1) make sure you re-compute partition each time
        ln_Z = self.partition_with_factor_weights(factor_weights)
        
        # 2) Define the LL:
        LL = sum([factor_weights[i]*factor_count for i, factor_count in enumerate(self.factor_counts)])
        
        LL = LL - (ln_Z)
        
        return LL * -1
    
    def objective_function_gradient(self, factor_weights):
        # Normal situation to get ln_z
        ln_Z = self.partition_with_factor_weights(factor_weights)
        
        exp_counts = []
        for i, i_z in enumerate(self.get_indicators()):
            expected_count_f = self.expected_count(i_z)
            exp_counts.append(expected_count_f)
            
        exp_counts = [math.exp(ece - ln_z) for ece in exp_counts]
        grad = [fc - ec for (fc,ec) in zip(self.factor_counts, exp_counts)]
        
        return grad
    
    def expected_count(self, indicator):
        wmc = self.sdd.wmc()
        previous = wmc.literal_weight(-indicator)
        self.wmc.set_literal_weight(-indicator, wmc.zero_weight)
        exp = wmc.propagate()
        self.wmc.set_literal_weight(-indicator, previous)
        return exp
    
    def partition_with_factor_weights(self, factor_weights):
        
        wmc = self.sdd.wmc()
        indicators = self.get_indicators()
        
        # Hacky: make sure only the actually used variables are considered in calculation
        for i in self.all_indicators():
           wmc.set_literal_weight(i, wmc.zero_weight )
        
        if len(factor_weights) != len(indicators):
            assert("Problem")
            
        # We now have a indicators <-> weights mappng
        for (i, w) in zip(indicators, factor_weights):
            wmc.set_literal_weight(i, w)
            #print(f"Set: {i} to {w}")
            
        return wmc.propagate()
            
    def count(self, factors, worlds):
        return [f.count(worlds)/len(worlds) for f in factors]
        
    # ------------------Add/Remove----------------
        
    def add_literal_factor(self, literal, weight=0):
        if not literal.is_literal():
            print("Problemo")
            
        self.factors.append((literal, weight, literal, literal.val() )) 
        
        if self.dynamic_update:                     # Only when dynamically updating we should change the SDD
            self.update_sdd()
        
    def add_factor(self, factor, weight=0):
        if not self.can_add_factors():
            print(f"Cannot add anymore factors.")
            return
        
        enc, ind = self.encode_factor(factor)
        self.factor_stack.append((factor, weight, enc, ind))  # Append the factor to the stack
        
        if self.dynamic_update:                     # Only when dynamically updating we should change the SDD
            self.update_sdd()
            
        self.nb_factors += 1
            
    def remove_factor(self, factor):
        if not self.factor_present(factor):
            return
            
        self.remove_factor_sdd_indicator(factor[3])    # TODO: remove hard coding
        self.nb_factors -= 1
        
    def remove_factor_sdd_indicator(self, indicator):
        
        if self.ind_is_literal(indicator):
            return
            
        self.pop_factor_indicator(indicator)                              
        self.condition_on_factor(indicator)           
        self.free_variable(indicator)        
        self.dirty = True
        
    def remove_factor_sdd_index(self, index):
    
        indicator_for_factor = self.get_factor_by_index(factor)       
        self.remove_factor_sdd_indicator(indicator_for_factor)
        
    def pop_factor_indicator(self, indicator):
        #TODO: efficiency
        factor = self.factor_by_indicator(indicator)
        self.factors = [(f, w, e, i) for (f, w, e, i) in self.factors if i != indicator] 
        return factor
        
    def factor_by_indicator(self, indicator):
        factors_by_ind = [(f, w, e, i) for (f, w, e, i) in self.factors if i == indicator]
        
        if len(factors_by_ind) == 0:
            return None
            
        if len(factors_by_ind) > 1:
            print(f"Found multiple factors with indicator {indicator}")
            
        return factors_by_ind[0]
        
    def condition_on_factor(self, indicator):
        # (F_1 <=> I_1 & F_2 <=> I_2) | (F_1 <=> I_1 & F_2 <=> -I_2) == F_1 <=> I_1
        # Essentially removes a factor without recmpiling
        l = self.mgr.condition(indicator,  self.sdd)
        r = self.mgr.condition(-indicator, self.sdd)
        self.sdd = self.mgr.disjoin(l,r)
        
    def update_sdd(self):   
        self.check_manager()                                    # Check if manager is ok
                
        if self.sdd == None:                                   # Initially set theory to True
            self.sdd = self.mgr.true()                         # Just true
        
        if len(self.factor_stack) > 0:                         # If there are any new factors then
            self.dirty = True                                  # Things will matter
        
        times = {"0":0,"1":0,"2":0,"3":0}
        
        for (f, w, e, i) in self.factor_stack:
            t1 = time()
            
            indicator = None
            encoded_factor = f
            
            t2 = time()
            
            left = self.sdd                                    # Current SDD
            right = e.compile(self.mgr)                         # Compile new factor to SDD
            
            t3=time()
            self.sdd = left & right           # Do the actual conjoining here
            t4 = time()
            
            self.factors.append((f, w, e, i)) # Add to the official collection
                                                               # Save (f, w, enc) encoding contains compiled SDD
            t5 = time()
            
            times["0"] += t2-t1
            times["1"] += t3-t2
            times["2"] += t4-t3
            times["3"] += t5-t4
            
        print(times)
        
        self.factor_stack = []
        
        return self.sdd
        
    def encode_factor(self, f):
        next_var = self.claim_next_available_variable()
        
        Pi = lit(next_var)              # Get indicator var
        encoding = equiv([f,Pi])        # Create encoding
        
        return encoding, next_var
        
    def can_add_factors(self):
        return self.nb_factors < self.max_nb_factors and self.indicator_manager.has_next()
    
    def claim_next_available_variable(self):        
        return self.indicator_manager.claim_next_available_variable()

    def free_variable(self, indicator):
        self.indicator_manager.free_variable(indicator)
        
    def check_manager(self):           
        if self.mgr == None:
            print("Cannot compile without a manager")
            assert()
        
    def validate_factor(self, factor):
        return True
        
    def set_indicator_manager(self, indicator_manager):
        if self.nb_factors > 0:
            assert("Cannot change indicator_manager after already using indicators from another manager")
            
        self.indicator_manager = indicator_manager
        
    def set_manager(self, manager=None):
        if manager == None:
            manager = SddManager(self.domain_size)
        self.mgr = manager
        self.dynamic_update = True
        
    def ind_is_literal(self, indicator):
        return indicator <= self.domain_size
        
    def feat_is_literal(self, feat):
        return self.ind_is_literal(feat[3])  # TODO:make not hard-coded.
        
    def factor_present(self, factor):
        return factor in self.factors
    
    # ------------------ Information ------------------------
    
    def is_literal_index(self, index):
        return index + 1 <= self.domain_size
    
    def set_factor_weights(self, weights):
        self.factors = [(f, nw, e, i) for ((f,_,e,i), nw) in zip(self.factors, weights)]
    
    def get_factor_by_index(self, index):
        return self.factors[index][3] #Hard coded indicator car is 3 #TODO: make not hard-coded.
    
    def get_factor_weights(self):
        # gets only the weights of the encoded factors
        return [w for (_, w, _, _) in self.factors]
    def get_weights(self):
        # gets all the weights(included literal weights)
        return self.theory_encoder.get_weights()
    
    def get_weights_array(self):
        return self.theory_encoder.weights_as_array()

    def get_encodings_and_weights(self):
        return [(e,w) for (_, w, e, _) in self.factors]
    
    def get_factors(self):
        return [f for (f, _, _, _) in self.factors]
    
    def get_encodings(self):
        return [e for (_, _, e, _) in self.factors]
    
    def get_indicators(self):
        return [i for (_, _, _, i) in self.factors]
    
    def get_stack_factors(self):
        return [f for (f, _) in self.factor_stack]
    
    def get_fwi(self):
        return [(f,w,i) for (f,w,_,i) in self.factors]
    
    def get_iw(self):
        return [(i,w) for (_,w,_,i) in self.factors]
    
    def get_w(self):
        return [w for (_, w, _, _) in self.factors]
        
    def get_features(self):
        return list(filter(lambda e : not self.feat_is_literal(e), self.factors))

    def all_indicators(self):
        return range(self.domain_size + 1, self.mgr.var_count() + 1)
    
    def available_pool(self):
        base = self.domain_size
        return []
        
    def sdd_size(self):
        if self.sdd == None:
            return 0
        return self.sdd.count()
    # ------------------- Evaluation -------------------------
    
    def evaluate(self, world):
        for (factor, _, _, _) in self.factors:
            if factor.evaluate(world) == False:
                return False
        return True
    
    def partition(self):
        if self.sdd == None:
            assert("Cannot perform counting without SDD")
        if len(self.factor_stack) > 0:
            print("Be carefull: there are still uncompiled factors in the model.")
        if not self.dirty: # If not dirty don't re-compute anything
            return self.Z
        
        
        wmc = self.sdd.wmc()        
        wmc = self.set_wmc_weights(wmc)
        
        self.Z = wmc.propagate()
        self.probs = self.probabilities(wmc)
        self.dirty = False         
        
        return self.Z
    
    def set_wmc_weights(self, wmc_manager):
        for i in self.all_indicators():
           wmc_manager.set_literal_weight(i, wmc_manager.zero_weight )
            
        for (i, w) in self.get_iw():
            wmc_manager.set_literal_weight(i,w)
            
        return wmc_manager
    
    def probabilities(self, wmc):
    
        indicators = self.get_indicators()
        encodings = self.get_encodings()
        
        self.probs = list(map(lambda x: wmc.literal_pr(x), indicators))            
        
        return [(e.to_string(),math.exp(p)) for (e,p) in zip(encodings, self.probs)]
        
    def LL(self, worlds):
        ln_Z = self.partition()
        
        weights = self.get_w()
        counts = self.count(self.get_factors(), worlds)
        ln_Z = self.Z
        
        self.ll = self.ll_calc(weights, counts, ln_Z)
        
        return self.ll
    
    def ll_calc(self, w, c, ln_Z):
        ll = sum([w_e * c_e for (w_e, c_e) in zip(w, c)]) 
        ll = ll - ln_Z
        return ll
    
    # ___ Printing Utils ___
    
    def to_string(self, mode="all"):
        if mode=="non-enc":
            ret = self.pretty_print_str(self.get_factors())
        elif mode == "stack":
            ret = self.pretty_print_str(self.get_stack_factors())
        elif mode == "all":
            ret = self.pretty_print_all()
        else:
            ret = self.pretty_print_str(self.get_encodings())
        
        return ret
    
    def pretty_print_all(self):
        pp =  "__Model__\n"
        pp += f"->  {self.domain_size} variables\n"
        pp += f"->  {len(self.factors)} factors compiled\n"
        pp += f"->  {len(self.factor_stack)} factors pending\n"
        pp += "Compiled factors:\n"
        pp += self.pretty_print_table(self.get_fwi())
        pp += "Uncompiled factors:\n"
        pp += self.pretty_print_table(self.factor_stack)
        #pp += "Weights of literals:\n"
        #pp += self.pretty_weights()
        
        if self.dirty:
            pp += "The model is dirty right now!\n"
        if self.Z != None:
            pp += "Probabilities:\n"
            pp += self.pretty_probs()
        return pp
    
    def pretty_print_table(self, table):
        # table in format: [(t_1, ..., t_n)]
        
        # 1) split into columns
        split_lists = self.table_to_columns(table)
        
        # 2) Space out strings
        split_lists = [self.space_out_strings(l) for l in split_lists]
    
        # 3) for easyness just go back to [(t,...,t)] format
        table_str = zip(*split_lists)
    
        # 4) Build table string
        table_str = [" ".join(table_row) for table_row in table_str]
        table_str = "\n".join(table_str)
        
        return table_str + "\n"
    
    def table_to_columns(self, table):
        if len(table) == 0:
            return []
        
        lists = []
        n_l = len(table[0])
        for i in range(n_l):
            lists.append(self.split_pairs(table, i))
            
        return lists
        
    def space_out_pairs(self, string_pairs):
        left = self.split_pairs(string_pairs, 0)
        right= self.split_pairs(string_pairs, 1)
        
        left = self.space_out_strings(left)
        right= self.space_out_strings(right)
        
        return list(zip(left, right))
    
    def split_pairs(self, pairs, i):
        return [x[i] for x in pairs]
    
    def space_out_strings(self, strings):
        strings = [str(x) for x in strings]
        max_l = max([len(x) for x in strings])
        return [self.space_out_string(x, max_l) for x in strings]
    
    def space_out_string(self, string, max_l):
        return string + (" " * (max_l - len(string)))
        
    def pretty_probs(self):
        st = map(lambda x: f"Factor: {x[0]} Prob: {x[1]}", self.space_out_pairs(self.probs))
        return "\n".join(st)
        
    def pretty_weights(self):
        mapped_w =  map(lambda v_w: f"W: {v_w[1]} V: {v_w[0]}", self.get_weights().items())
        str_w = reduce(lambda x, y: x + "\n" + y, list(mapped_w))
        return str_w
        
    def pretty_print_str(self, facts):
        if len(facts) == 0:
            return ""
        f_strings = map(lambda x:x.to_string(), facts)
        return reduce(lambda x,y: x+ "\n" + y, f_strings) + "\n"
    
    def pretty_encodings_and_weights(self):
        e_w = self.get_encodings_and_weights()
    
