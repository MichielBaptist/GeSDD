import numpy as np
import pysdd
import math
import random

import matplotlib.pyplot as plt
import sklearn
import utils.time_utils as tu

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
from utils.time_utils import timer


from model.indicator_manager import indicator_manager

class Model:
    """
    Class representing a logical Theory, just a conjunction of Rules. Each rule has a weight.
    """

    # --------------------- init stuff --------------------
    def __init__(self,
                domain_size,
                manager = None,
                indicator_manager = None,
                count_manager = None,
                max_nb_factors=20,
                unique_ID = None):
        self.factors = []              # List of (formula, weight, encoding, indicator)
        self.domain_size = domain_size # How many literals can this theory have?
        self.nb_factors = 0
        self.max_nb_factors = max_nb_factors
        self.unique_ID = unique_ID

        # All managers
        self.count_manager = count_manager
        self.indicator_manager = indicator_manager
        self.mgr = manager
        self.sdd = None

        self.add_initial_literals()           # Add the initial factors which give weight to literals A, B, C ....

        # Nothing at start
        self.Z = None
        self.probs = None
        self.dirty = False                    # Is there need for recomputation?

        #if self.indicator_manager == None:
        #    self.indicator_manager = self.init_indicator_manager(self.domain_size, self.max_nb_factors)

    def free(self):
        #print("starting to free model")
        self.sdd = None
        self.mgr = None
        self.cmgr = None
        self.nb_factors = None
        self.dirty = None
        self.Z = None
        self.probs = None

        #for (_,_,_,i) in self.get_features():
        #    self.indicator_manager.free_variable(i)

        self.indicator_manager = None
        self.factors = None
        self.domain_size = None
        #print("Done freeing model")

    def add_initial_literals(self):
        for i in range(self.domain_size):
            self.add_literal_factor(lit(i + 1))

    def init_indicator_manager(self, domain_size, max_number_factors):
        bottom = domain_size + 1
        top = bottom + max_number_factors
        return indicator_manager(range(bottom, top))

    def set_compiled_sdd(self, sdd):
        # Method assumes sdd is refed already!
        # Derefs the old one.

        if sdd == self.sdd:
            return

        """
        if self.sdd != None and (not self.sdd.garbage_collected() and self.sdd.ref_count() != 0):
            #print(f"Current sdd is still reffed: {self.sdd.ref_count()}")
            for i in range(self.sdd.ref_count()):
                self.sdd.deref()"""

        self.sdd = sdd
        self.dirty = True

    def from_pre_compiled_sdd(self, sdd, factors):
        if self.sdd != None:
            print("Model has sdd already!")
        if self.nb_factors > 0:
            print("Model has factors already!")

        self.factors = factors
        self.nb_factors = len(factors)
        self.set_compiled_sdd(sdd)

    """
    Note: We can just copy the SDD since the SDD is an immutable
    structure. Applying
        - conjunctions
        - disjunctions
        - conditioning
        - ...
    Returns a new SDD, it does not alter the current SDD.

    Also all logical rules are considered immutable. And thus Only
    the list has to be coppied, the elements may be references.
    """
    def clone(self):

        clone = self.soft_clone()

        #for (_,_,_,i) in clone.get_features():
        #    clone.indicator_manager.increment_variable(i)

        return clone

    def soft_clone(self):

        clone = Model(self.domain_size)

        clone.indicator_manager = self.indicator_manager
        clone.count_manager = self.count_manager
        clone.mgr = self.mgr
        clone.sdd = self.sdd
        clone.factors = list(self.factors)
        clone.nb_factors = self.nb_factors
        clone.max_nb_factors = self.max_nb_factors
        clone.dirty = True
        clone.Z = None
        clone.probs = []
        clone.unique_ID = self.unique_ID

        return clone

    def set_unique_ID(self, unique_ID):
        self.unique_ID = unique_ID

    # ------------------ Training ----------------

    def fit(self, worlds, set_id, acc = 5e-2, max_fun=15):
        # worlds: [[{1,0}]]

        # 1) Count the worlds for each factor in the SDD:
        self.factor_counts = self.count(self.get_factors(), worlds, set_id)
        self.number_worlds = len(worlds)

        self.wmc = self.sdd.wmc()

        # 2) define the objective
        objective = self.objective_function

        # 3) define the gradient of the objective
        gradient = self.objective_function_gradient

        # 4) Optimize
        res = minimize(objective, x0 = self.get_factor_weights(), jac = None, method="BFGS",
                      options={
                            #'gtol': acc,
                            'disp': False,
                            'maxiter': max_fun
                            })

        # 5) Set the weights correct!
        self.set_factor_weights(res.x)

        self.dirty = True
        self.wmc = None
        self.mgr.set_prevent_transformation(False)

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

    def count(self, factors, worlds, set_id):
        if (self.count_manager == None):
            print("Count manager not present! Highly recommended to set one!")
            return [sum([f.evaluate(world) for world in worlds])/len(worlds) for f in factors]

        return self.count_manager.count_factors(factors, worlds, set_id)


    # ------------------Add/Remove----------------

    """
    This method adds a literal factor. A literal factor is
    a logical rule consisting of only a literal. The indicator
    of a literal factor is also the literal i.e. l <=> l
    The weight of the factor is initially 0.

    This method is used at the initialization of a model. Thus
    initializing results in a markov network with no edges.
    """
    def add_literal_factor(self, literal, weight=0):

        if not literal.is_literal():
            #print("Trying to add a literal factor which is not a literal...")
            return

        self.factors.append((literal, weight, literal, literal.val() ))

    """
    This method adds a compiled factor to the model. A compiled
    factor should have the following form: (f, w, e, i)
        - f : Logical rule.
        - w : Weight of this factor.
        - e : The encoding of the factor. This MUST be present.  e = (f <=> i)
        - i : The indicator of the compiled factor. i.e.  i in   e = (f <=> i)
    """
    def add_compiled_factor(self, factor):
        # factor: (f, w, e, i)

        (f, w, e, i) = factor
        if self.indicator_present(i):
            print(f"""Cannot add factor {e}. The indicator is already used by\n
            the following factor: {self.factor_by_indicator(i)}""")
            return

        # When this factor is used, we should increment the indicator
        self.indicator_manager.increment_variable(factor[3])
        self.factors.append(factor)
        self.nb_factors += 1

    """
    Add a non-compiled factor to the model.
        - factor: a logical factor
        - weight: an initial weight
    """
    def add_factor(self, factor, weight=0):

        # Cannot add anymore factors
        if not self.can_add_factors():
            print("Trying to add a factor over the allowed limit.")
            return None

        # Already present
        if self.feature_present(factor):
            print("Trying to add factor already present.")
            return None

        enc, ind = self.encode_factor(factor)

        # Can never happen according to the encoding
        if self.indicator_present(ind):
            print("Trying to add a factor with the same indicator.")
            return None

        # Actually add the factor
        self.factors.append((factor, weight, enc, ind))

        self.nb_factors += 1
        self.dirty = True

        return (factor, weight, enc, ind)

    def remove_factor(self, factor):
        # factor: (f, w, e, i)

        if not self.factor_present(factor):
            print("Tried to remove a factor that is not present")
            return

        # Removing automatically "frees" the indicator
        self.factors.remove(factor)

        # Free indicator
        #self.free_variable(factor[3])

        self.nb_factors -= 1
        self.dirty = True

    # Compiles this model based on the currently
    # present factors.
    def compile(self):
        # Note: conj automatically ref()s the top level sdd.
        #       it doesn't ref any intermediate results and
        #       will be garbage collected.

        # 1) manager should be ok
        self.check_manager()

        # 2) The SDD is then just the conjunction of the rules
        if len(self.get_feature_encodings()) == 0:
            conjunction = cons(True)
        else:
            conjunction = conj(self.get_feature_encodings())

        sdd = conjunction.to_sdd(self.mgr)
        self.set_compiled_sdd(sdd)

        return self.sdd

    def encode_factor(self, f):

        # 1) get next indicator variable
        next_var = self.claim_next_available_variable()

        # 2) encode factor
        Pi = lit(next_var)              # Get indicator var
        encoding = equiv([f,Pi])        # Create encoding

        return encoding, next_var

    def can_add_factors(self):
        return len(self.get_factors()) < self.mgr.var_count()
        #return self.indicator_manager.has_next()

    def claim_next_available_variable(self):
        next_var = self.domain_size + 1
        while next_var in self.get_indicators():
            next_var = next_var + 1

        return next_var

    #def claim_next_available_variable(self, factor):
    #    return self.indicator_manager.claim_next_available_variable(factor)

    #def free_variable(self, indicator):
    #    self.indicator_manager.free_variable(indicator)

    def check_manager(self):
        if self.mgr == None:
            print("Cannot compile without a manager")
            assert()

    def set_count_manager(self, count_manager):
        self.count_manager = count_manager

    def set_indicator_manager(self, indicator_manager):
        if self.nb_factors > 0:
            assert("Cannot change indicator_manager after already using indicators from another manager")

        self.indicator_manager = indicator_manager

    def set_manager(self, manager=None):
        if manager == None:
            manager = SddManager(self.domain_size)
        self.mgr = manager

    def ind_is_literal(self, indicator):
        return indicator <= self.domain_size

    def feat_is_literal(self, feat):
        return self.ind_is_literal(feat[3])  # TODO:make not hard-coded.

    def factor_present(self, factor):
        return factor in self.factors

    def feature_present(self, feature):
        return feature in self.get_f()

    def indicator_present(self, indicator):
        return indicator in self.get_indicators()

    # ------------------ Information ------------------------

    def is_literal_index(self, index):
        return index + 1 <= self.domain_size

    def set_factor_weights(self, weights):
        self.factors = [(f, nw, e, i) for ((f,_,e,i), nw) in zip(self.factors, weights)]
        self.dirty = True

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

    def get_feature_encodings(self):
        return [e for (_, _, e, _) in self.get_features()]

    def get_indicators(self):
        return [i for (_, _, _, i) in self.factors]

    def get_fwi(self):
        return [(f,w,i) for (f,w,_,i) in self.factors]

    def get_iw(self):
        return [(i,w) for (_,w,_,i) in self.factors]

    def get_w(self):
        return [w for (_, w, _, _) in self.factors]

    def get_f(self):
        return [f for (f,_,_,_) in self.factors]

    def get_features(self):
        return list(filter(lambda e : not self.feat_is_literal(e), self.factors))

    def all_indicators(self):
        return range(self.domain_size + 1, self.mgr.var_count() + 1)

    def sdd_size(self):
        if self.sdd == None:
            return 0

        return self.sdd.size()
    # ------------------- Evaluation -------------------------

    def world_probability(self, world):
        ln_Z = self.partition()

        # Way 1
        w_sum = 0
        for (f, w, i) in self.get_fwi():
            if f.evaluate(world):
                w_sum += w

        return math.exp(w_sum - ln_Z)

    def evaluate(self, world):
        for (factor, _, _, _) in self.factors:
            if factor.evaluate(world) == False:
                return False
        return True

    def partition(self):
        if self.sdd == None:
            assert("Cannot perform counting without SDD")
        if not self.dirty and self.Z != None: # If not dirty don't re-compute anything
            return self.Z

        wmc = self.sdd.wmc()
        wmc = self.set_wmc_weights(wmc)

        self.Z = wmc.propagate()
        self.probs = self.probabilities(wmc)
        self.dirty = False

        self.mgr.set_prevent_transformation(False)

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

    def LL(self, worlds, set_id):
        ln_Z = self.partition()

        weights = self.get_w()
        counts = self.count(self.get_factors(), worlds, set_id)

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
        elif mode == "all":
            ret = self.pretty_print_all()
        else:
            ret = self.pretty_print_str(self.get_encodings())

        return ret

    def pretty_print_all(self):
        pp =  f"__Model {self.unique_ID}__\n"
        pp += f"->  {self.domain_size} variables\n"
        pp += f"->  {len(self.factors)} factors compiled\n"
        pp += "Compiled factors:\n"
        pp += self.pretty_print_table([("Factor", "Weight", "Indicator")] + self.get_fwi())
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
