from model.model import Model
from pysdd.sdd import SddNode

from logic.conj import conj

import numpy as np
import math

class crossover:
    def apply(self, model1, model2) -> (Model, Model, dict):
        pass

    def applyOneSDD(self, model, args) -> SddNode:
        pass

class rule_trade_cross(crossover):

    def __init__(self,
                 run_name,
                 nb_rules_min = 1,
                 nb_rules_max = 5,
                 nb_rules_pct = 0.2,):
        self.nb_rules_min = nb_rules_min
        self.nb_rules_max = nb_rules_max
        self.nb_rules_pct = nb_rules_pct
        self.run_name = run_name

    def __str__(self):
        lines = [
            "Rule trade crossiver",
            f"-->PCT: {self.nb_rules_pct}"
        ]
        return "\n".join(lines)

    def apply(self, left, right):
        # left: model
        # right: model

        largs = {
            "feat":None
        }
        rargs = {
            "feat":None
        }
        args = {
            "largs": largs,
            "rargs": rargs
        }

        # One can now assume the children have seperate managers.
        lchild = hard_clone(left, self.run_name)
        rchild = hard_clone(right, self.run_name)

        # Find best rule to swap
        lfeats = lchild.get_features()
        rfeats = rchild.get_features()

        nb_left = math.ceil(self.nb_rules_pct * len(lfeats))
        nb_right= math.ceil(self.nb_rules_pct * len(rfeats))

        lfeats, rfeats = list_difference(lfeats, rfeats)

        nb_left = min(nb_left, len(lfeats))
        nb_right= min(nb_right, len(rfeats))

        lweights = [abs(w) for (f,w,_,_) in lfeats]
        probs = softmax_weights(lweights)
        #print(f"Tradable left rules: {lfeats}")
        if nb_left > 0:
            selected_indices = np.random.choice(
                range(len(lfeats)),
                size= nb_left,
                replace=False,
                p = probs)
        else:
            selected_indices = []
        lselect = [lfeats[i] for i in selected_indices]
        #print(f"Selected left rules: {nb_left} --> {lselect}")

        rweights = [0.7*abs(w) - 0.3*math.log2(len(f.all_conjoined_literals())) for (f,w,_,_) in rfeats]
        rweights = [abs(w) for (f,w,_,_) in rfeats]
        probs = softmax_weights(rweights)
        if nb_right > 0:
            selected_indices = np.random.choice(
                range(len(rfeats)),
                size= nb_right,
                replace=False,
                p = probs)
        else:
            selected_indices = []
        rselect = [rfeats[i] for i in selected_indices]
        #print(f"Tradable right rules: {rfeats}")
        #print(f"Selected right rules: {nb_right} --> {rselect}")

        new_right_features = [rchild.add_factor(f, w) for (f,w,_,_) in lselect]
        new_left_features = [lchild.add_factor(f, w) for (f,w,_,_) in rselect]

        new_left_features = [e for e in new_left_features if e is not None]
        new_right_features= [e for e in new_right_features if e is not None]

        rargs["feat"] = new_right_features
        largs["feat"] = new_left_features

        return (lchild, rchild, args)

        """
        lselect = None
        rselect = None

        if len(lfeats) > 0:
            lweights = [abs(w) for (_,w,_,_) in lfeats]
            probs = softmax_weights(lweights)
            selected_index = np.random.choice(range(len(lfeats)), replace = False, p = probs)
            lselect = lfeats[selected_index]

        if len(rfeats) > 0:
            rweights = [abs(w) for (_,w,_,_) in rfeats]
            probs = softmax_weights(rweights)
            selected_index = np.random.choice(range(len(rfeats)), replace = False, p = probs)
            rselect = rfeats[selected_index]

        # Add the selected features
        if lselect != None:
            rargs["feat"] = rchild.add_factor(lselect[0], lselect[1])

        if rselect != None:
            largs["feat"] = lchild.add_factor(rselect[0], rselect[1])

        return (lchild, rchild, args)
        """

    def applyOneSDD(self, sdd, mgr, args):
        # sdd: referenced sdd
        # mgr: manager of the sdd
        # args: "feat": the feature to add to this sdd. None if
        #               none should be added

        if len(args["feat"]) == 0:
            return sdd

        new_feats = conj([enc for (_,_,enc,_) in args["feat"]])

        # feat_sdd is referenced by .to_sdd(mgr)
        feats_sdd = new_feats.to_sdd(mgr)

        # New sdd is simply conjunction
        new_sdd = feats_sdd & sdd

        # input "sdd" is also referenced at this point but should not
        # be dereferenced in this method.
        new_sdd.ref()
        feats_sdd.deref()

        return new_sdd

class subset_cross(crossover):

    def __init__(self,
                 subset_pct = 0.2):
        self.subset_pct = subset_pct

    def apply(self, left, right):
        # left: model
        # right: model

        largs = {
            "feat":None,
            "mdl":left.to_string()
        }
        rargs = {
            "feat":None,
            "mdl":right.to_string()
        }
        args = {
            "largs": largs,
            "rargs": rargs
        }

        # One can now assume the children have seperate managers.
        lchild = hard_clone(left, self.run_name)
        rchild = hard_clone(right, self.run_name)

        # Find best rule to swap
        lfeats = lchild.get_features()
        rfeats = rchild.get_features()

        nb_left = math.ceil(self.subset_pct * len(lfeats))
        nb_right= math.ceil(self.subset_pct * len(rfeats))

        lfeats, rfeats = list_difference(lfeats, rfeats)

        nb_left = min(nb_left, len(lfeats))
        nb_right= min(nb_right, len(rfeats))

        lweights = [0.7*abs(w) - 0.3*math.log2(len(f.all_conjoined_literals())) for (f,w,_,_) in lfeats]
        lweights = [abs(w) for (f,w,_,_) in lfeats]
        probs = softmax_weights(lweights)
        probs = None
        #print(f"Tradable left rules: {lfeats}")
        if nb_left > 0:
            selected_indices = np.random.choice(
                range(len(lfeats)),
                size= nb_left,
                replace=False,
                p = probs)
        else:
            selected_indices = []
        lselect = [lfeats[i] for i in selected_indices]
        #print(f"Selected left rules: {nb_left} --> {lselect}")

        rweights = [0.7*abs(w) - 0.3*math.log2(len(f.all_conjoined_literals())) for (f,w,_,_) in rfeats]
        rweights = [abs(w) for (f,w,_,_) in rfeats]
        probs = softmax_weights(rweights)
        probs = None
        if nb_right > 0:
            selected_indices = np.random.choice(
                range(len(rfeats)),
                size= nb_right,
                replace=False,
                p = probs)
        else:
            selected_indices = []
        rselect = [rfeats[i] for i in selected_indices]
        #print(f"Tradable right rules: {rfeats}")
        #print(f"Selected right rules: {nb_right} --> {rselect}")

        for rr in rselect:
            rchild.remove_factor(rr)
        for lr in lselect:
            lchild.remove_factor(lr)

        new_right_features = [rchild.add_factor(f, w) for (f,w,_,_) in lselect]
        new_left_features = [lchild.add_factor(f, w) for (f,w,_,_) in rselect]

        new_left_features = [e for e in new_left_features if e is not None]
        new_right_features= [e for e in new_right_features if e is not None]

        rargs["new"] = new_right_features
        rargs["old"] = rselect
        largs["new"] = new_left_features
        largs["old"] = lselect

        return (lchild, rchild, args)

        """
        lselect = None
        rselect = None

        if len(lfeats) > 0:
            lweights = [abs(w) for (_,w,_,_) in lfeats]
            probs = softmax_weights(lweights)
            selected_index = np.random.choice(range(len(lfeats)), replace = False, p = probs)
            lselect = lfeats[selected_index]

        if len(rfeats) > 0:
            rweights = [abs(w) for (_,w,_,_) in rfeats]
            probs = softmax_weights(rweights)
            selected_index = np.random.choice(range(len(rfeats)), replace = False, p = probs)
            rselect = rfeats[selected_index]

        # Add the selected features
        if lselect != None:
            rargs["feat"] = rchild.add_factor(lselect[0], lselect[1])

        if rselect != None:
            largs["feat"] = lchild.add_factor(rselect[0], rselect[1])

        return (lchild, rchild, args)
        """

    def applyOneSDD(self, sdd, mgr, args):
        # sdd: referenced sdd
        # mgr: manager of the sdd
        # args: "feat": the feature to add to this sdd. None if
        #               none should be added

        new_sdd = self.remove_old(sdd, mgr, args)
        new_sdd = self.add_new(new_sdd, mgr, args)

        return new_sdd

    def remove_old(self, sdd, mgr, args):
        if len(args["old"]) == 0:
            return sdd

        indicators = [ind for (_,_,_,ind) in args["old"]]
        return remove_indicators(sdd, mgr, indicators)

    def add_new(self, sdd, mgr, args):

        if len(args["new"]) == 0:
            return sdd

        new_feats = conj([enc for (_,_,enc,_) in args["new"]])

        # feat_sdd is referenced by .to_sdd(mgr)
        feats_sdd = new_feats.to_sdd(mgr)

        # New sdd is simply conjunction
        new_sdd = feats_sdd & sdd

        # input "sdd" is also referenced at this point but should not
        # be dereferenced in this method.
        new_sdd.ref()
        feats_sdd.deref()

        return new_sdd

def hard_clone(model, pth):
    clone = model.clone()

    # Copies the Manager
    clone.mgr = model.mgr.copy([])
    clone.mgr.auto_gc_and_minimize_on()

    # Cpies the sdd ==> is there another way?
    #clone.sdd = model.sdd.copy(manager = clone.mgr)
    clone.sdd = copy_sdd(clone.mgr, model.sdd, pth)
    clone.sdd.ref()

    return clone

def copy_sdd(mgr, sdd, pth):
    tmp_save_path = f"ga/crossover/tmp/tmp_sdd_{pth}.sdd"
    save_sdd(sdd, tmp_save_path)
    copy = load_sdd(mgr, tmp_save_path)

    return copy

def save_sdd(sdd, tmp_path):
    sdd.save(tmp_path.encode())

def load_sdd(mgr, tmp_path):
    return mgr.read_sdd_file(tmp_path.encode())

def softmax_weights(weights):
    p = [math.exp(w) for w in weights]
    norm = sum(p)
    return [w/norm for w in p]

def list_difference(l1, l2):
    # Removes the common elements from list 1 and list 2
    l1_filtered = list(l1)
    l2_filtered = list(l2)
    for i in l1:
        for j in l2:
            if position_equal(i, j, [0]): #TODO: Ew ew ew, rework
                #print("equality")
                l1_filtered.remove(i)
                l2_filtered.remove(j)

    return l1_filtered, l2_filtered

def position_equal(l1, l2, pos):
    # Checks if the array is equal at the positions indexed by pos
    return [l1[i] for i in pos] == [l2[i] for i in pos]

def remove_indicators(sdd, mgr, indicators):
    # sdd: an sdd with exactly one reference count
    #print(f"Manager live Before threshold: {mgr.live_size()}")
    #print(f"Removing indicators: {indicators}")
    #print(f"Input sdd has a ref count of: {sdd.ref_count()}")

    if len(indicators) == 0:
        # remains on a 1 reference count
        return sdd

    input_sdd = sdd
    input_sdd.ref()
    out_sdd = None
    for ind in indicators:

        out_sdd = remove_indicator(input_sdd, mgr, ind)
        irb = input_sdd.ref_count()
        orb = out_sdd.ref_count()

        if input_sdd == out_sdd:
            print("Input == output")

        if input_sdd != out_sdd:
            input_sdd.deref()

        ira = input_sdd.ref_count()
        ora = out_sdd.ref_count()

        input_sdd = out_sdd

        #print(f"input ref:  {irb}-->{ira}")
        #print(f"output ref: {orb}-->{ora}")

    #print(f"Manager live After threshold: {mgr.live_size()}")
    #print(f"Input SDD size: {sdd.size()}")
    #print(f"Output SDD size: {out_sdd.size()}")

    return out_sdd

def remove_indicator(sdd, mgr, indicator):
    # sdd: assumed to be a referenced sdd and
    # new_sdd: becomes referenced aswell

    l = mgr.condition(indicator, sdd)
    l.ref()
    r = mgr.condition(-indicator, sdd)
    r.ref()

    new_sdd = l | r
    new_sdd.ref()
    l.deref()
    r.deref()

    return new_sdd
