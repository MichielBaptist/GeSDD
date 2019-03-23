import pysdd
from gen.gen import recursive_generator
from model.model import Model
import random
import numpy as np
import time
from functools import reduce

from logic.conj import conj
from logic.disj import disj
from logic.equiv import equiv
from logic.cons import cons
from logic.lit import lit
from logic.neg import neg

import math

import utils.string_utils as stru

class cross_over:
    def apply(self, model1, model2) -> Model:
        pass

class rule_swapping_cross(cross_over):
    def __init__(self, rules_transferred = 1):
        self.rules_transferred = rules_transferred

    def apply(self, left, right):
        # No need to make new sdd's for this cross over!

        # Get rules
        left_rules = left.get_features()
        right_rules = right.get_features()
        # Remove the common rules (no use transferring these)
        left_rules, right_rules = list_difference(left_rules, right_rules)

        # None untill found
        left_transfer_rule = None
        right_transfer_rule = None

        # Only if there are enough rules to transfer
        if len(left_rules) >= self.rules_transferred:
            # pick a rules based on softmax
            weights = [w for (_,w,_,_) in left_rules]
            probs = softmax(weights)
            left_transfer_rule = select_random_element(left_rules, probs)

        # Only if there are enough rules to transfer
        if len(right_rules) >= self.rules_transferred:
            # pick a rules based on softmax
            weights = [w for (_,w,_,_) in right_rules]
            probs = softmax(weights)
            right_transfer_rule = select_random_element(right_rules, probs)

        # Copy the models completely
        new_left = left.clone()
        new_right = right.clone()

        if left_transfer_rule != None:
            new_right.add_compiled_factor(left_transfer_rule)

        if right_transfer_rule != None:
            new_left.add_compiled_factor(right_transfer_rule)

        return new_left, new_right

    def __str__(self):
        lines = [
            "Rule swapping",
            f"--> Nb. Rules swapped: {self.rules_transferred}",
            f"--> Weighting: Softmax"
        ]
        return "\n".join(lines)


class simple_subset_cross(cross_over):
    def apply(self, left, right):

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

        # Remove the previous models
        left.free()
        right.free()

        # Return the new models
        return new_left, new_right

    def __str__(self):
        lines = [
            "Siple Subset cross"
        ]
        return "\n".join(lines)

class mutation:
    def apply(self, model) -> Model:
        pass

class sign_flipping_mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model):

        if model.nb_factors == 0:
            return model

        n = select_random_element([1,2])
        random_rule = select_random_element(model.get_features())

        #for random_rules in random_rules:
        model.remove_factor(random_rule)

        (f, _, _, _) = random_rule

        new_rule = self.flip_sign(f, model)

        model.add_factor(new_rule, 0)

        return model

    def flip_sign(self, rule, model):
        return self.flip_sign1(rule, model)

    def flip_sign1(self, rule, model):

        if isinstance(rule, conj):
            rule = self.flip_conjunction(rule, model)
        elif isinstance(rule, disj):
            print("disj")
        elif isinstance(rule, equiv):
            print("equiv")

        return rule

    def flip_conjunction(self, rule, model):
        conjunctors = rule.list_of_factors

        random_flip = select_random_element(conjunctors)

        if isinstance(random_flip, neg):
            flipped_conjunctor = random_flip.negated_factor
        else:
            flipped_conjunctor = neg(random_flip)

        new_conjunctors = [c for c in conjunctors if c != random_flip] + [flipped_conjunctor]

        return conj(new_conjunctors)


class rule_shrinking_mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model):

        if model.nb_factors == 0:
            return model

        random_rule = select_random_element(model.get_features())
        model.remove_factor(random_rule)

        (f, _, _, _) = random_rule

        new_rule = self.shrink_rule(f, model)

        model.add_factor(new_rule, 0)

        return model

    def shrink_rule(self, f, model):
        return self.shrink_rule1(f, model)


    """
    Simplest shrink rule... Just choose a random subtree
    """
    def shrink_rule1(self, rule, model):

        if isinstance(rule, conj):
            rule = self.shrink_conjunction(rule, model)
        elif isinstance(rule, disj):
            print("disj")
        elif isinstance(rule, equiv):
            print("equiv")

        return rule


    """
        heuristics:
            - based on weight, maybe find the lowest weight/size (sdd or conjunctions)
            - based on corrolation matrix?
            - ...
    """
    def shrink_conjunction(self, rule, model):
        # 1) find set
        conjunctors = rule.list_of_factors


        if len(conjunctors) <= 2:
            return rule

        removed_conjunctor = select_random_element(conjunctors)

        new_conjunctors = [c for c in conjunctors if c != removed_conjunctor]

        if len(new_conjunctors) == 0:
            print(f"removed conjunctor: {removed_conjunctor.to_string()}")
            print(f"conjunction: {rule.to_string()}")

        return conj(new_conjunctors)

    def __str__(self):
        lines = [
            "Rule shrinkage mutation",
            "a1 & ... & an-1 & an => a1 & ... & an-1"
        ]

        return "\n".join(lines)

class rule_expansion_mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model):
        if model.nb_factors == 0:
            return model

        # Doesn't include literals
        random_rule = select_random_element(model.get_features())
        model.remove_factor(random_rule)

        # --- actually alter the rule here ----
        # random_rule: (f, w, e, i)
        # --> f: logical factors    (will use this as base)
        # --> w: weight             (not used)
        # --> e: encoding           (not used)
        # --> i: indicators         (not used)

        (f, _, _, _) = random_rule

        # This method is subjetc to change
        new_rule = self.expand_rule(f, model)

        # Weight is random for now
        model.add_factor(new_rule, 0)

        return model

    def expand_rule(self, rule, model):
        # rule: logical factor, contains the SDD in rule.to_sdd() cached
        return self.expand_rule1(rule, model)

    """
    Rule expansion based on randomly picking a new variable
    not based on any information on the data e.g. corrolation
    or mutual information or entropy
    Ideas:
        -corrolation matrix based weighting of model_selection
        -mutual information based
        -literal weight based selection (add low weight/ high weight)
    """
    def expand_rule1(self, rule, model):
        # Conjoin random variable
        domain_size = model.domain_size

        if isinstance(rule, conj):
            # 1) Get literals already in conjunction
            list_of_factors = rule.list_of_factors
            list_of_literals = list(filter(lambda x: x.is_literal(), list_of_factors))

            # 2) Get remaining available literals
            unavailable_literals = rule.all_conjoined_literals()
            available_literals = [l for l in range(1, domain_size+1) if l not in unavailable_literals
                        and -l not in unavailable_literals ]

            if len(available_literals) == 0:
                return rule

            # 3) Finally select a new literal to join the conjuntion
            random_literal = select_random_element(available_literals) * select_random_element([1,-1])
            new_rule =  conj([rule, lit(random_literal)])

        elif isinstance(rule, disj):
            print("disj")
        elif isinstance(rule, equiv):
            print("equiv")

        return new_rule

    def __str__(self):
        lines = [
        "Rule-expansion mutation",
        "a1 ° ... ° an => a1 ° ... ° an ° an+1"
        ]

        return "\n".join(lines)

class add_mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model):

        # Domain size
        domain_size = model.domain_size

        gen = recursive_generator(domain_size, None, None)
        (random_f, random_w) = gen.random_feature()

        model.add_factor(random_f, random_w)

        return model

    def __str__(self):
        lines = [
            "Add mutation",
            f"--> Rule generaion: generator(domain_size).random_feature()"
        ]
        return "\n".join(lines)

class remove_mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model):

        if model.nb_factors == 0:
            return model

        features = model.get_features()
        random_rem = select_random_element(features)

        model.remove_factor(random_rem)

        return model

    def __str__(self):
        lines = [
            "Remove mutation",
            f"--> Rule removel: uniform"
        ]
        return "\n".join(lines)

class weighted_remove_mutation(mutation):
    def __init__(self):
        self.max_exponent = 500
        pass

    def apply(self, model):

        if model.nb_factors == 0:
            return model

        features = model.get_features()
        importance = [math.exp(min(1/abs(w), self.max_exponent)) for (_,w,_,_) in features]
        probabilities = [i/sum(importance) for i in importance]

        random_rem = select_random_element(features, probabilities)

        model.remove_factor(random_rem)

        return model

    def __str__(self):
        lines = [
            "Remove mutation",
            f"--> Rule removel: softmax"
        ]
        return "\n".join(lines)

class threshold_mutation(mutation):
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, individual):
        features = individual.get_features()
        bad_features = [(f, w, i, e) for (f, w, i, e) in features if abs(w) <= self.threshold]

        #print(individual.to_string())
        #print(bad_features)

        for feature in bad_features:
            individual.remove_factor(feature)

        return individual

    def __str__(self):
        lines = [
            "Threshold mutation",
            f"--> Threshold: {self.threshold}"
        ]
        return "\n".join(lines)

class script_mutation(mutation):
    def __init__(self, mutations):
        self.mutations = mutations

    def apply(self, individual):
        for mutation in self.mutations:
            individual = mutation.apply(individual)
        return individual

    def __str__(self):
        lines = [
            "Script mutation"
        ]
        mut_str = [(str(i+1) + ")",str(mut)) for i, mut in enumerate(self.mutations)]
        mut_str = stru.pretty_print_table(mut_str)
        lines += [mut_str]
        return "\n".join(lines)

class multi_mutation(mutation):
    def __init__(self, mutations, distr=None):
        self.mutations = mutations
        self.mutation_distribution = distr

        if self.mutation_distribution == None:
            self.mutation_distribution = [1/len(self.mutations) for i in range(len(self.mutations))]

    def apply(self, individual):
        #print(self.mutations)
        mutation = select_random_element(self.mutations, self.mutation_distribution)
        return mutation.apply(individual)

    def __str__(self):
        lines = [
            "Multi mutation"
        ]
        mut_str = [("Option", "Probability", "Mutation")]
        mut_str += [(i+1, self.mutation_distribution[i], str(mut)) for i, mut in enumerate(self.mutations)]
        mut_str = stru.pretty_print_table(mut_str)
        lines += [mut_str]
        return "\n".join(lines)

class selection:
    def apply(self, pop, fitness, n):
        # pop and fitness should be equal size, where each pair formed by (pop[i], fitness[i])
        # should be the fitness of individual i
        pass

# Implements basic weighted selection
# Except there is a certain regularizer
class Weighted_selection(selection):
    def __init__(self, reg):
        self.regularizer = reg

    def apply(self, pop, fitness, n):
        pop_n = len(pop)
        probs = [fi/sum(fitness) for fi in fitness]
        selected_ind = np.random.choice(range(pop_n), replace = False, p=probs, size = n)
        non_selected_ind = [i for i in range(pop_n) if i not in selected_ind]

        s_pop =  take_indices(pop, selected_ind)
        s_fit =  take_indices(fitness, selected_ind)
        ns_pop = take_indices(pop, non_selected_ind)
        ns_fit = take_indices(fitness, non_selected_ind)

        return (s_pop, s_fit), (ns_pop, ns_fit)


    def __str__(self):
        lines = [
            "Weighted selection without regularizer",
            "--> P(i) = wi / Z"
        ]
        return "\n".join(lines)

class pairing:
    def pair(self, pop, fit):
        zipped = zip(pop, fit)
        sorted_pop = sorted(zipped, key = lambda x : x[1], reverse=True)

        pairs = []
        for i in range(0, len(sorted_pop), 2):
            pairs.append((sorted_pop[i][0], sorted_pop[i+1][0]))

        return pairs

    def __str__(self):
        return "Standard ordered pairing"


def list_difference(l1, l2):
    # Removes the common elements from list 1 and list 2
    l1_filtered = list(l1)
    l2_filtered = list(l2)
    for i in l1:
        for j in l2:
            if position_equal(i, j, [0,2,3]): #TODO: Ew ew ew, rework
                #print("equality")
                l1_filtered.remove(i)
                l2_filtered.remove(j)

    return l1_filtered, l2_filtered

def position_equal(l1, l2, pos):
    # Checks if the array is equal at the positions indexed by pos
    return [l1[i] for i in pos] == [l2[i] for i in pos]

def select_random_element(elements, distr=None):
    index = np.random.choice(len(elements), replace=False, p = distr)
    return elements[index]

def empty_child(source):
    # TODO: transfer weights???
    domain_size = source.domain_size
    imgr = source.indicator_manager
    mgr = source.mgr

    empty = Model(domain_size, imgr, mgr)
    empty.dynamic_update = True
    return empty

def split_collection(collection, distr):

    n_sets = len(distr)
    n_coll = len(collection)
    distr = [i/sum(distr) for i in distr]
    sets = [[] for i in range(n_sets)]

    choices = np.random.choice(range(n_sets), n_coll, distr)

    for (c, f) in zip(choices, collection):
        sets[c].append(f)

    #print([len(s) for s in sets])
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

def take_indices(arr, ind):
    return [e for i, e in enumerate(arr) if i in ind]

def softmax(w):
    exp = [math.exp(i) for i in w]
    z = sum(exp)
    return [i/z for i in exp]
