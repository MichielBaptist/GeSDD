from pysdd.sdd import SddNode
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

class mutation:
    # Applies the mutation to the model. This method does not change the SDD
    # it only changes the model (primitive datatypes only). These mutations
    # should be efficient and will be run in series. This is because some
    # behavior cannot be captured in parallel programming, operations such
    # as changing a common data structure (i.e. indicator manager).
    # This method should return 2 things:
    #       1) The changed model
    #       2) A dict of arguments needed to chenge the SDD
    #          this dict may not contain any structures which are not
    #          pickleable. This dict will be given to applySDD Method.
    def apply(self, model, data) -> (Model, dict):
        pass

    # Changes actual SDD (potentially in parallel). This method should always
    # return an SDD which is referenced. This is in order to prevent premature
    # garbage collection. This method should usually not dereference the
    # input SDD except in some exceptional cases.
    def applySDD(self, sdd, mgr, args) -> SddNode:
        pass

    # Method to dereference the input sdd given that the output sdd
    # is actually different.
    def applySDD_wrap(self, sdd, mgr, args):
        new_sdd = self.applySDD(sdd, mgr, args)

        if new_sdd != sdd:
            sdd.deref()

        return new_sdd

class feat_shrink__mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model, data):

        args = {"changed":False}

        feats = model.get_features()
        feats = [feat for feat in feats if len(feat[0].list_of_factors) > 2]

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        feat = select_random_element(feats)

        # 0: feature
        # 1: weight
        # 2: encoding
        # 3: indicator
        ft = feat[0]

        shrunk_feat = self.shrink_feat(ft)

        if shrunk_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(shrunk_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args

    def applySDD(self, sdd, mgr, args):

        if not args["changed"]:
            return sdd

        #print(args)

        #1) remove the old factor
        removed_ind = args["old"][3]
        trimmed_sdd = remove_indicator(sdd, mgr, removed_ind)

        # 2) create the new feature, already encoded!
        new_feat = args["new"][2]
        feat_sdd = new_feat.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def shrink_feat(self, feat):
        if isinstance(feat, conj):

            # 1) get all conjunctors
            conjunctors = feat.list_of_factors

            # 2) select random subset from conjunctors

            # Size of subset
            k = select_random_element(list(range(2, len(conjunctors))))
            # select subset
            subset = np.random.choice(range(len(conjunctors)), size = k, replace=False )
            #print(subset)
            subset = [conjunctors[i] for i in sorted(subset)]
            #print(subset)

            print(f"{feat} --> {conj(subset)}")

            return conj(subset)
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

class feat_expand__mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model, data):

        args = {"changed":False}

        feats = model.get_features()

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        feat = select_random_element(feats)

        # 0: feature
        # 1: weight
        # 2: encoding
        # 3: indicator
        ft = feat[0]

        expand_feat = self.expand_feat(ft, model.domain_size)

        if expand_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(expand_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args

    def applySDD(self, sdd, mgr, args):

        if not args["changed"]:
            return sdd

        #print(args)

        #1) remove the old factor
        removed_ind = args["old"][3]
        trimmed_sdd = remove_indicator(sdd, mgr, removed_ind)

        # 2) create the new feature, already encoded!
        new_feat = args["new"][2]
        feat_sdd = new_feat.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def expand_feat(self, feat, domain_size):
        def to_lit(v):
            if v < 0:
                return neg(lit(abs(v)))
            else:
                return lit(v)

        if isinstance(feat, conj):

            # 1) get all conjunctors (also negative)
            conjunctors_val = feat.all_conjoined_literals()
            conjunctors_abs = [abs(v) for v in conjunctors_val]

            conjunctors = feat.list_of_factors

            # 2) select new literals to add to feat
            remaining_literals = [l for l in list(range(1, domain_size + 1)) if l not in conjunctors_abs]

            if len(remaining_literals) == 0:
                return feat

            # 3) sample random additions
            add_n = min(int(3*random.random()), len(remaining_literals))
            #print(remaining_literals)
            #print(conjunctors_val)
            #print(conjunctors)

            additions = list(np.random.choice(remaining_literals, replace = False, size=add_n))

            new_conjunctors = conjunctors_val + additions
            new_conjunctors = sorted(new_conjunctors, key=abs)

            new_conj = conj([to_lit(con) for con in new_conjunctors])

            #print(conjunctors + additions)
            #print(additions.__repr__())
            #print(conjunctors.__repr__())
            #print(f"{feat} --> {new_conj}")

            return new_conj
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

class individual_sign_flipping_mutation(mutation):
    def __init__(self, ratio=0.2):
        self.negation_ratio = ratio
        pass

    def apply(self, model, data):

        args = {"changed":False}

        feats = model.get_features()

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        feat = select_random_element(feats)

        # 0: feature
        # 1: weight
        # 2: encoding
        # 3: indicator
        ft = feat[0]

        negated_feat = self.negate_operands_randomly(ft, self.negation_ratio)

        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args

    def applySDD(self, sdd, mgr, args):

        if not args["changed"]:
            return sdd

        #print(args)

        #1) remove the old factor
        removed_ind = args["old"][3]
        trimmed_sdd = remove_indicator(sdd, mgr, removed_ind)

        # 2) create the new feature, already encoded!
        new_feat = args["new"][2]
        feat_sdd = new_feat.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def negate_operands_randomly(self, feat, ratio):
        if isinstance(feat, conj):
            return conj([self.negate_pr(c, ratio) for c in feat.list_of_factors])
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

    def negate_pr(self, f, p):
        negate = random.random() < p
        if negate and isinstance(f, neg):
            return f.negated_factor
        elif negate and not isinstance(f, neg):
            return neg(f)
        elif not negate:
            return f

class sign_flipping_mutation(mutation):
    def __init__(self, ratio=0.5):
        self.negation_ratio = ratio
        pass

    def apply(self, model, data):

        print("-----------Flipping global ----------")
        args = {"changed":False}

        feats = model.get_features()
        feats = [f for f in feats if len(f[0].list_of_factors) > 2]

        print(f"Remaining feats: {feats}")

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        feat = select_random_element(feats)

        # 0: feature
        # 1: weight
        # 2: encoding
        # 3: indicator
        ft = feat[0]

        negated_feat = self.negate_operands_randomly(ft, self.negation_ratio)

        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args

    def applySDD(self, sdd, mgr, args):

        if not args["changed"]:
            return sdd

        #print(args)

        #1) remove the old factor
        removed_ind = args["old"][3]
        trimmed_sdd = remove_indicator(sdd, mgr, removed_ind)

        # 2) create the new feature, already encoded!
        new_feat = args["new"][2]
        feat_sdd = new_feat.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def negate_operands_randomly(self, feat, ratio):
        if isinstance(feat, conj):

            # 1) get all conjunctors
            conjunctors = feat.list_of_factors

            # 2) select random subset from conjunctors

            # Size of subset
            k = select_random_element(list(range(2, len(conjunctors))))
            # select subset
            subset = np.random.choice(range(len(conjunctors)), size = k, replace=False )
            subset = [conjunctors[i] for i in subset]


            # filter out the rest of conjunctors
            not_subset = [c for c in conjunctors if not c in subset]

            # 3) create a new conjunction
            if len(not_subset) == 0:
                new_feat = neg(conj(subset))
            else:
                new_feat = conj(not_subset + [neg(conj(subset))])

            print(f"New generated feature: {feat} -- > {new_feat}")

            return new_feat
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

class add_n_mutation(mutation):
    def __init__(self, gen, min_n = 1, max_n = 5):
        self.gen = gen
        self.min_n = min_n
        self.max_n = max_n

    def apply(self, model, data):
        # setup args
        args = {}

        # 1) sample an amount of features to add
        n = select_random_element(range(self.min_n, self.max_n+1))

        # 2) generate all new features
        feats = [self.gen.random_feature() for i in range(n)]

        # 3) add all features
        feats = [model.add_factor(f,w) for (f,w) in feats]

        # 4) remove all features which could not be added to the model
        #    (doesn't affect model)
        feats = [e for e in feats if e is not None]

        # Nothing changed
        if len(feats) == 0:
            return model, {"changed":False}
        else:
            # all added feats
            print(f"Adding {len(feats)} new feats!")
            return model, {"changed":True, "added":feats}

    def applySDD(self, sdd, mgr, args):
        # args["changed"]: did the model chage at all?
        # args["added"]: all added rules


        if not args["changed"]:
            return sdd

        # Each new rule is a conjunction so can add all at the same time
        new_rules = conj([encoding for (_,_,encoding,_) in args["added"]])

        # new_rule referenced at top level only
        # referencing the new rule happens in .to_sdd()
        new_rules = new_rules.to_sdd(mgr)

        # new_sdd needs to be referenced again
        new_sdd = sdd & new_rules

        # Reference the new SDD
        new_sdd.ref()

        # dereference new_rule for auto gc and minimize.
        # The new rule is an intermediate step so should be derefed.
        new_rules.deref()

        return new_sdd

class remove_n_mutation(mutation):
    def __init__(self, min_n = 1, max_n = 5):
        self.min_n = min_n
        self.max_n = max_n

    def apply(self, model, data):

        # 1) Select a random rule to be removed (based on weight).
        feats = model.get_features()

        if len(feats) == 0:
            return model, {"changed": False}

        new_max_n = min(len(feats), self.max_n)

        n = select_random_element(range(self.min_n, new_max_n+1))
        removed_feats = list(np.random.choice(range(len(feats)), size = n, replace = False ))
        removed_feats = [feats[i] for i in removed_feats]

        #print(model.to_string())

        for feat in removed_feats:
            model.remove_factor(feat)

        #print(model.to_string())

        print(f"removing {len(removed_feats)} feats!")
        return model, {"changed":True, "removed":removed_feats}


    def applySDD(self, sdd, mgr, args):
        # sdd: ref()'d SDD
        # mgr: manager of SDD
        # args: "removed_ind" = indicator of removed rule.
        #print(args)

        if not args["changed"]:
            return sdd

        #print(args)

        indicators = [ind for (_,_,_,ind) in args["removed"]]
        return remove_indicators(sdd, mgr, indicators)

class replace_n_mutation(mutation):
    def __init__(self, add_n, rem_n):
        self.add_n = add_n
        self.rem_n = rem_n

    def apply(self, model, data):
        mdl, rem_args = self.rem_n.apply(model)
        mdl, add_args = self.add_n.apply(mdl)
        return mdl, {"rargs":rem_args, "aargs":add_args}

    def applySDD(self, sdd, mgr, args):
        # After this: both sdd and trimmed are reffed
        trimmed = self.rem_n.applySDD(sdd, mgr, args["rargs"])
        # After this: trimmed no longer reffed, input sdd and output sdd reffed
        new_sdd = self.add_n.applySDD_wrap(trimmed, mgr, args["aargs"])

        return new_sdd

class threshold_mutation(mutation):

    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, model, data = None):
        # setup args
        trim_list = []
        args = {"trim":trim_list}

        # 1) Find all the factors which are below a threshold
        facts = model.get_features()
        remove_list = [fact for fact in facts if abs(fact[1]) < self.threshold]

        # 2) remove all trimmed factors from the model
        for fact in remove_list:
            model.remove_factor(fact)
            trim_list.append(fact[3])

        return model, args

    def applySDD(self, sdd, mgr, args):
        # sdd: expected to be referenced SDD already.

        # Nothing changes ==> just return SDD
        # Thus output SDD is already reffed
        if len(args["trim"]) == 0:
            return sdd

        # Else remove each indicator in the trimmed list
        # no heuristic is employed to choose the indicators
        new_sdd = remove_indicators(sdd, mgr, args["trim"])

        return new_sdd

    def __str__(self):
        return f"Threshold mutation \n Threshold: {self.threshold}"

class remove_mutation(mutation):
    def __init__(self):
        pass

    def apply(self, model, data):

        # Setup args
        args = {}

        # 1) Select a random rule to be removed (based on weight).
        rules = model.get_features()

        if len(rules) == 0:
            return model, {"changed": False}

        args["changed"] = True
        weights = [-abs(w) for (_, w, _, _) in rules]
        selected_index = np.random.choice(range(len(rules)), replace = False, p = softmax_weights(weights))
        selected_rule = rules[selected_index]

        model.remove_factor(selected_rule)
        args["removed_ind"] = selected_rule[3]

        return model, args


    def applySDD(self, sdd, mgr, args):
        # sdd: ref()'d SDD
        # mgr: manager of SDD
        # args: "removed_ind" = indicator of removed rule.
        #print(args)

        if not args["changed"]:
            return sdd

        return remove_indicator(sdd, mgr, args["removed_ind"])

class add_mutation(mutation):

    def __init__(self, generator):
        self.generator = generator

    def apply(self, model, data):
        # setup args
        args = {}

        # 1) Generate a new feature
        feat_f, feat_w = self.generator.random_feature()

        # 2) Add the feature to the model
        #    without altering the SDD just yet
        #    this method will take care of: - encoding
        #                                   - indicator stuff
        #                                   - ...
        new_fct = model.add_factor(feat_f, feat_w)

        if new_fct == None:
            return model, {"changed":False}
        else:
            (f,w,e,i) = new_fct
            args["f"] = f
            args["w"] = w
            args["i"] = i
            args["e"] = e
            args["changed"]= True

            return model, args

    def applySDD(self, sdd, mgr, args):
        # args["changed"]: did the model chage at all?
        # args["f"]: feature that was added
        # args["w"]: weight of feature
        # args["i"]: indicator of added feature
        # args["e"]: enciding of added feature


        if not args["changed"]:
            return sdd

        # Want to compile the encoding of the rule
        e = args['e']

        # new_rule referenced at top level only
        # referencing the new rule happens in .to_sdd()
        new_rule = e.to_sdd(mgr)

        # new_sdd needs to be referenced again
        new_sdd = sdd & new_rule

        # Reference the new SDD
        new_sdd.ref()

        # dereference new_rule for auto gc and minimize.
        # The new rule is an intermediate step so should be derefed.
        new_rule.deref()

        return new_sdd

class script_mutation(mutation):
    def __init__(self, script):
        # script: [mutation]
        self.script = script

    def apply(self, model, data):
        # Here we have to apply several mutations. We can do this by
        # first applying all the mutation affecting the models and collecting
        # all the sdd arguments. in applySDD we ca then iteratively apply them
        # to the SDDs.

        individual = model
        args_list = []
        for mutation in self.script:
            individual, args = mutation.apply(individual, data)
            args_list.append(args)

        return individual, {"args_list":args_list}


    def applySDD(self, sdd, mgr, args):
        # sdd: ref()'d SDD
        # mgr: manager of SDD
        # args: "args_list" = list of args
        args_list = args["args_list"]

        # Assuming sdd obtained from mutation is always ref()'d .
        # In accordance to the above specification. The wrap Method should take
        # care of dereffing intermediate results.
        new_sdd = sdd
        for args, mutation in zip(args_list, self.script):
            new_sdd = mutation.applySDD_wrap(new_sdd, mgr, args)

        return new_sdd

    def applySDD_wrap(self, sdd, mgr, args):
        return self.applySDD(sdd, mgr, args)

    def __str__(self):
        tbl = [("Nb.", "Mutation")]
        tbl = tbl + [(i, m) for i, m in enumerate(self.script)]
        return stru.pretty_print_table(tbl, "Script mutation")

class multi_mutation(mutation):
    def __init__(self, mutations, distr=None):
        self.mutations = mutations
        self.distr = distr

        if self.distr == None:
            self.distr = [1/len(mutations) for i in mutations]

        pass

    def apply(self, model, data):
        args = {}
        args["selected_mutation"] = np.random.choice(self.mutations, p = self.distr)

        res_mdl, res_args = args["selected_mutation"].apply(model, data)

        args["res_args"] = res_args

        return res_mdl, args

    def applySDD(self, sdd, mgr, args):
        # args: "selected_mutation": selected mutation
        #       "res_args": args of mutation application
        return args["selected_mutation"].applySDD(sdd, mgr, args["res_args"])

    def __str__(self):
        lines = [
            "Multi mutation"
        ]
        mut_str = [("Option", "Probability", "Mutation")]
        mut_str += [(i+1, self.distr[i], str(mut)) for i, mut in enumerate(self.mutations)]
        mut_str = stru.pretty_print_table(mut_str)
        lines += [mut_str]
        return "\n".join(lines)

class identity_mutation(mutation):

    def __init__(self):
        pass

    def apply(self, model, data):
        return model, None

    def applySDD(self, sdd, mgr, args):
        return sdd

class feat_expand_pct_mutation(mutation):
    def __init__(self,
                 pct = 0.2):
        self.pct = pct
        pass

    def apply(self, model, data):
        print("------------Expansion---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        # 2) negate
        fts = [self.expand_feat(ft[0], model.domain_size) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From -------> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm} -----> {to}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print("Old ---------> New")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm} --> {to}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

        """
        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args
        """

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def expand_feat(self, feat, domain_size):
        def to_lit(v):
            if v < 0:
                return neg(lit(abs(v)))
            else:
                return lit(v)

        if isinstance(feat, conj):

            # 1) get all conjunctors (also negative)
            conjunctors_val = feat.all_conjoined_literals()
            conjunctors_abs = [abs(v) for v in conjunctors_val]

            conjunctors = feat.list_of_factors

            # 2) select new literals to add to feat
            remaining_literals = [l for l in list(range(1, domain_size + 1)) if l not in conjunctors_abs]

            if len(remaining_literals) == 0:
                return feat

            # 3) sample random additions
            add_n = min(int(3*random.random()), len(remaining_literals))
            #print(remaining_literals)
            #print(conjunctors_val)
            #print(conjunctors)

            additions = list(np.random.choice(remaining_literals, replace = False, size=add_n))

            new_conjunctors = conjunctors_val + additions
            new_conjunctors = sorted(new_conjunctors, key=abs)

            new_conj = conj([to_lit(con) for con in new_conjunctors])

            #print(conjunctors + additions)
            #print(additions.__repr__())
            #print(conjunctors.__repr__())
            #print(f"{feat} --> {new_conj}")

            return new_conj
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

class remove_pct_mutation(mutation):
    def __init__(self, pct = 0.2):
        self.pct = pct

    def apply(self, model, data):

        # 1) Select a random rule to be removed (based on weight).
        feats = model.get_features()

        print("----------Removing---------")

        if len(feats) == 0:
            return model, {"changed": False}


        n = min(len(feats), math.ceil(len(feats) * self.pct))
        probs = [0.4*math.log2(len(f.all_conjoined_literals())) - 0.6*abs(w) for (f, w,_,_) in feats]
        probs = softmax_weights(probs)
        #probs = None
        tbl = [("Weight", "len()", "log(len())", "P(sel)")]
        tbl = tbl + list(zip([w for (f,w,_,_) in feats],
                                    [len(f.all_conjoined_literals()) for (f,_,_,_) in feats],
                                    [math.log2(len(f.all_conjoined_literals())) for (f,_,_,_) in feats],
                                    probs))
        print(stru.pretty_print_table(tbl, top_bar=True, bot_bar=True))
        #print(tbl)

        removed_feats = list(np.random.choice(range(len(feats)), size = n, replace = False, p=probs ))
        removed_feats = [feats[i] for i in removed_feats]

        #print(model.to_string())

        for feat in removed_feats:
            model.remove_factor(feat)

        #print(model.to_string())

        print(f"removing {len(removed_feats)} feats!")
        for f in removed_feats:
            print(f)

        return model, {"changed":True, "removed":removed_feats}


    def applySDD(self, sdd, mgr, args):
        # sdd: ref()'d SDD
        # mgr: manager of SDD
        # args: "removed_ind" = indicator of removed rule.
        #print(args)

        if not args["changed"]:
            return sdd

        #print(args)

        indicators = [ind for (_,_,_,ind) in args["removed"]]
        return remove_indicators(sdd, mgr, indicators)

    def __str__(self):
        lines = [
            "Percentage remove mutation"
        ]
        tbl = [("Prop", "Val")]
        tbl = tbl + [("PCT", self.pct), ("Remove heuristic", "softmax(-|wi|)")]
        lines.append(stru.pretty_print_table(tbl))
        return "\n".join(lines)

class feat_shrink_pct_mutation(mutation):
    def __init__(self,
                 pct = 0.2):
        self.pct = pct
        pass

    def apply(self, model, data):
        print("------------Shrinking---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()
        feats = [f for f in feats if len(f[0].list_of_factors) > 2]

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        print(selected_feats)

        # 2) negate
        fts = [self.shrink_feat(ft[0]) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From -------> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm} -----> {to}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print("Old ---------> New")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm} --> {to}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

        """
        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args
        """

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def shrink_feat(self, feat):
        if isinstance(feat, conj):

            # 1) get all conjunctors
            conjunctors = feat.list_of_factors

            # 2) select random subset from conjunctors
            # Size of subset
            k = select_random_element(list(range(2, len(conjunctors))))
            # select subset
            subset = np.random.choice(range(len(conjunctors)), size = k, replace=False )
            #print(subset)
            subset = [conjunctors[i] for i in sorted(subset)]
            #print(subset)

            return conj(subset)
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

class feat_flip_pct_mutation(mutation):
    def __init__(self,
                 pct = 0.2,
                 ratio=0.5):
        self.negation_ratio = ratio
        self.pct = pct
        pass

    def apply(self, model, data):
        print("------------Flipping---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        # 2) negate
        fts = [self.negate_operands_randomly(ft[0], self.negation_ratio) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From -------> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm} -----> {to}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print("Old ---------> New")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm} --> {to}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

        """
        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args
        """

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def negate_operands_randomly(self, feat, ratio):
        if isinstance(feat, conj):
            return conj([self.negate_pr(c, ratio) for c in feat.list_of_factors])
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

    def negate_pr(self, f, p):
        negate = random.random() < p
        if negate and isinstance(f, neg):
            return f.negated_factor
        elif negate and not isinstance(f, neg):
            return neg(f)
        elif not negate:
            return f

class add_pct_mutation(mutation):
    def __init__(self, gen, pct = 0.15):
        self.gen = gen
        self.pct = pct

    def apply(self, model, data):
        # setup args
        args = {}

        print("--------------Adding-----------")

        # 1) sample an amount of features to add
        n = math.ceil(self.pct * len(model.get_features()))

        # 2) generate all new features
        feats = [self.gen.random_feature() for i in range(n)]

        # 3) add all features
        feats = [model.add_factor(f,w) for (f,w) in feats]

        # 4) remove all features which could not be added to the model
        #    (doesn't affect model)
        feats = [e for e in feats if e is not None]

        # Nothing changed
        if len(feats) == 0:
            return model, {"changed":False}
        else:
            # all added feats
            print(f"Adding {len(feats)} new feats:")
            for f in feats:
                print(f)
            return model, {"changed":True, "added":feats}

    def applySDD(self, sdd, mgr, args):
        # args["changed"]: did the model chage at all?
        # args["added"]: all added rules


        if not args["changed"]:
            return sdd

        # Each new rule is a conjunction so can add all at the same time
        new_rules = conj([encoding for (_,_,encoding,_) in args["added"]])

        # new_rule referenced at top level only
        # referencing the new rule happens in .to_sdd()
        new_rules = new_rules.to_sdd(mgr)

        # new_sdd needs to be referenced again
        new_sdd = sdd & new_rules

        # Reference the new SDD
        new_sdd.ref()

        # dereference new_rule for auto gc and minimize.
        # The new rule is an intermediate step so should be derefed.
        new_rules.deref()

        return new_sdd

class apmi_feat_expand_pct_mutation(mutation):
    def __init__(self,
                 pct = 0.2,
                 add_mx = 2,
                 k = 3,
                 set_id = "train"):
        self.pct = pct
        self.k = k
        self.add_mx = add_mx
        self.set_id = set_id
        pass

    def apply(self, model, data):
        print("------------Expansion---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))
        n = select_random_element(list(range(1,n+1)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        # 2) expand
        fts = [self.expand_feat(
                    ft[0],
                    model.domain_size,
                    self.k,
                    model.count_manager,
                    data) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From --> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm[0]} --> {None if to == None else to[0]}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print(f"Old --> New --> {len(new_feats)}")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm[0]} --> {to[0]}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

        """
        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args
        """

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def expand_feat(self, feat, domain_size, k, cmgr, train_dat):


        if isinstance(feat, conj):

            # 1) get all conjunctors (also negative)
            conjunctors_val = feat.all_conjoined_literals()
            conjunctors_abs = [abs(v) for v in conjunctors_val]

            conjunctors = feat.list_of_factors

            # 2) select new literals to add to feat
            remaining_literals = [l for l in list(range(1, domain_size + 1)) if l not in conjunctors_abs]

            if len(remaining_literals) == 0:
                return feat

            # 3) sample random additions
            add_n = min(math.ceil(self.add_mx*random.random()), len(remaining_literals))
            #print(remaining_literals)
            #print(conjunctors_val)
            #print(conjunctors)

            candidates = [self.generate_candidate(
                remaining_literals,
                add_n,
                conjunctors_val) for i in range(k)]

            apmis = find_apmis_conjunctions(candidates, train_dat, self.set_id, cmgr)
            probs = [e/sum(apmis) for e in apmis]

            tbl = [("candidate", "APMI", "Probs")]
            tbl = tbl + [(feat, find_apmi(feat.list_of_factors, train_dat, "train", cmgr), "//")]
            tbl = tbl + list(zip(candidates, apmis, probs))

            print(stru.pretty_print_table(tbl))

            #print(conjunctors + additions)
            #print(additions.__repr__())
            #print(conjunctors.__repr__())
            #print(f"{feat} --> {new_conj}")

            return select_random_element(candidates, distr=probs)
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

    def generate_candidate(self,
                            remaining_literals,
                            add_n,
                            conjunctors_val):
        def to_lit(v):
            if v < 0:
                return neg(lit(abs(v)))
            else:
                return lit(v)

        additions = list(np.random.choice(remaining_literals, replace = False, size=add_n))

        new_conjunctors = conjunctors_val + additions
        new_conjunctors = sorted(new_conjunctors, key=abs)

        new_conj = conj([to_lit(con) for con in new_conjunctors])

        return new_conj

    def __str__(self):
        lines = [
        ""
        ]
        tbl = [("Prop", "Val"),
            ("PCT", self.pct),
            ("K", self.k),
            ("Measure", "AMI"),
            ("Selection", "P(f) = AMI/ sum(AMI)")
        ]
        return stru.pretty_print_table(tbl, name = "AMI PCT expand mutation")

class apmi_feat_shrink_pct_mutation(mutation):
    def __init__(self,
                 pct = 0.2,
                 rm_mx = 2,
                 k = 3,
                 set_id = "train"):
        self.pct = pct
        self.k = k
        self.rm_mx = rm_mx
        self.set_id = set_id
        pass

    def apply(self, model, data):
        print("------------Shrinking---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()
        feats = [f for f in feats if len(f[0].list_of_factors) > 2]

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))
        n = select_random_element(list(range(1,n+1)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        # 2) expand
        fts = [self.shrink_feat(
                    ft[0],
                    self.k,
                    model.count_manager,
                    data) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From --> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm[0]} --> {None if to == None else to[0]}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print(f"Old --> New --> {len(new_feats)}")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm[0]} --> {to[0]}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

        """
        if negated_feat == ft:
            return model, args

        #print(model)

        # We know fore sure we can add a factor
        new_feat = model.add_factor(negated_feat, random.random())
        if new_feat == None: # means that we tried adding already present feature
            return model, args
        model.remove_factor(feat)

        #print(model)

        args["old"] = feat
        args["new"] = new_feat
        args["changed"] = True

        return model, args
        """

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def shrink_feat(self, feat, k, cmgr, train_dat):


        if isinstance(feat, conj):

            # 1) get all conjunctors
            conjunctors = feat.list_of_factors

            # 2) select random subset from conjunctors
            # Size of subset
            subset_n = select_random_element(list(range(2, len(conjunctors))))
            # select subset

            candidates = [self.generate_candidate(
                conjunctors,
                subset_n
                ) for i in range(k)]

            apmis = find_apmis_conjunctions(candidates, train_dat, self.set_id, cmgr)
            probs = [e/sum(apmis) for e in apmis]

            tbl = [("candidate", "APMI", "Probs")]
            tbl = tbl + [(feat, find_apmi(feat.list_of_factors, train_dat, "train", cmgr), "//")]
            tbl = tbl + list(zip(candidates, apmis, probs))

            print(stru.pretty_print_table(tbl))

            #print(conjunctors + additions)
            #print(additions.__repr__())
            #print(conjunctors.__repr__())
            #print(f"{feat} --> {new_conj}")

            return select_random_element(candidates, distr=probs)
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

    def generate_candidate(self,
                            conjunctors,
                            subset_n):
            subset = np.random.choice(range(len(conjunctors)), size = subset_n, replace=False )
            #print(subset)
            subset = [conjunctors[i] for i in sorted(subset)]
            #print(subset)

            return conj(subset)

    def __str__(self):
        tbl = [("Prop", "Val"),
            ("PCT", self.pct),
            ("K", self.k),
            ("Measure", "AMI"),
            ("Selection", "P(f) = AMI/ sum(AMI)")
        ]
        return stru.pretty_print_table(tbl, name = "AMI PCT shrink mutation")

class apmi_feat_flip_pct_mutation(mutation):
    def __init__(self,
                 pct = 0.2,
                 k = 3,
                 ratio = 0.5,
                 set_id = "train"):
        self.pct = pct
        self.k = k
        self.negation_ratio = ratio
        self.set_id = set_id
        pass

    def apply(self, model, data):
        print("------------Flipping---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))
        n = select_random_element(list(range(1,n+1)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        # 2) expand
        fts = [self.negate_operands_randomly(
                    ft[0],
                    self.negation_ratio,
                    self.k,
                    model.count_manager,
                    data) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From --> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm[0]} --> {None if to == None else to[0]}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print(f"Old --> New --> {len(new_feats)}")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm[0]} --> {to[0]}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def negate_operands_randomly(self, feat, ratio, k, cmgr, train_dat):


        if isinstance(feat, conj):

            candidates = [self.generate_candidate(
                feat,
                ratio
                ) for i in range(k)]

            apmis = find_apmis_conjunctions(candidates, train_dat, self.set_id, cmgr)
            probs = [e/sum(apmis) for e in apmis]

            tbl = [("candidate", "APMI", "Probs")]
            tbl = tbl + [(feat, find_apmi(feat.list_of_factors, train_dat, "train", cmgr), "//")]
            tbl = tbl + list(zip(candidates, apmis, probs))

            print(stru.pretty_print_table(tbl))

            #print(conjunctors + additions)
            #print(additions.__repr__())
            #print(conjunctors.__repr__())
            #print(f"{feat} --> {new_conj}")

            return select_random_element(candidates, distr=probs)
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

    def generate_candidate(self, feat, ratio):
        return conj([self.negate_pr(c, ratio) for c in feat.list_of_factors])

    def negate_pr(self, f, p):
        negate = random.random() < p
        if negate and isinstance(f, neg):
            return f.negated_factor
        elif negate and not isinstance(f, neg):
            return neg(f)
        elif not negate:
            return f

    def __str__(self):
        tbl = [("Prop", "Val"),
            ("PCT", self.pct),
            ("K", self.k),
            ("Measure", "AMI"),
            ("Ratio", self.negation_ratio),
            ("Selection", "P(f) = AMI/ sum(AMI)")
        ]
        return stru.pretty_print_table(tbl, name = "AMI PCT individual flip mutation")

class apmi_feat_flip_pct_mutation_global(mutation):
    def __init__(self,
                 pct = 0.2,
                 k = 3,
                 ratio = 0.5,
                 set_id = "train"):
        self.pct = pct
        self.k = k
        self.negation_ratio = ratio
        self.set_id = set_id
        pass

    def apply(self, model, data):
        print("------------Flipping Global---------------")

        args = {"changed":False, "old":[], "new":[]}

        feats = model.get_features()
        feats = [f for f in feats if len(f[0].list_of_factors) > 2]

        if len(feats) == 0:
            return model, args
        if not model.can_add_factors():
            return model, args

        # 1) select
        n = min(len(feats), math.ceil(self.pct * len(feats)))
        n = select_random_element(list(range(1,n+1)))

        feats_indices = np.random.choice(
            range(len(feats)),
            size = n,
            replace=False
        )
        selected_feats = [feats[i] for i in feats_indices]

        # 2) expand
        fts = [self.negate_operands_randomly(
                    ft[0],
                    self.negation_ratio,
                    self.k,
                    model.count_manager,
                    data) for ft in selected_feats]

        # 3) add
        neg_feats = [model.add_factor(neg_feat) for neg_feat in fts]

        print("From --> To")
        for (frm, to) in zip(selected_feats, neg_feats):
            print(f"{frm[0]} --> {None if to == None else to[0]}")

        # 4) remove old
        old_feats = []
        new_feats = []
        for (old_f, new_f) in zip(selected_feats, neg_feats):
            if new_f != None:
                model.remove_factor(old_f)
                old_feats.append(old_f)
                new_feats.append(new_f)

        print(f"Old --> New --> {len(new_feats)}")
        for (frm, to) in zip(old_feats, new_feats):
            print(f"{frm[0]} --> {to[0]}")

        args["old"] = old_feats
        args["new"] = new_feats
        args["changed"] = True

        return model, args

    def applySDD(self, sdd, mgr, args):

        if len(args["old"]) == 0:
            return sdd

        #print(args)

        #1) remove the old factors
        removed_indicators = [f[3] for f in args["old"]]
        trimmed_sdd = remove_indicators(sdd, mgr, removed_indicators)

        # 2) create the new feature, already encoded!
        added_feats = conj([f[2] for f in args["new"]])
        feat_sdd = added_feats.to_sdd(mgr)

        # 3) conjoin
        new_sdd = feat_sdd & trimmed_sdd

        feat_sdd.deref()
        trimmed_sdd.deref()
        new_sdd.ref()

        return new_sdd

    def negate_operands_randomly(self, feat, ratio, k, cmgr, train_dat):


        if isinstance(feat, conj):

            conjunctors = feat.list_of_factors
            subset_n = select_random_element(list(range(2, len(conjunctors))))

            candidates = [self.generate_candidate(
                ratio,
                conjunctors,
                subset_n
                ) for i in range(k)]

            apmis = find_apmis_conjunctions(candidates, train_dat, self.set_id, cmgr)
            probs = [e/sum(apmis) for e in apmis]

            tbl = [("candidate", "APMI", "Probs")]
            tbl = tbl + [(feat, find_apmi(feat.list_of_factors, train_dat, "train", cmgr), "//")]
            tbl = tbl + list(zip(candidates, apmis, probs))

            print(stru.pretty_print_table(tbl))

            #print(conjunctors + additions)
            #print(additions.__repr__())
            #print(conjunctors.__repr__())
            #print(f"{feat} --> {new_conj}")

            return select_random_element(candidates, distr=probs)
        else:
            print(f"Feat was not conjuntion!!! {feat}")
            return feat

    def generate_candidate(self, ratio, conjunctors, subset_n):
        subset = np.random.choice(range(len(conjunctors)), size = subset_n, replace=False )
        subset = [conjunctors[i] for i in subset]

        # filter out the rest of conjunctors
        not_subset = [c for c in conjunctors if not c in subset]

        # 3) create a new conjunction
        if len(not_subset) == 0:
            new_feat = neg(conj(subset))
        else:
            new_feat = conj(not_subset + [neg(conj(subset))])
        return new_feat

    def __str__(self):
        tbl = [("Prop", "Val"),
            ("PCT", self.pct),
            ("K", self.k),
            ("Measure", "AMI"),
            ("Ratio", self.negation_ratio),
            ("Selection", "P(f) = AMI/ sum(AMI)")
        ]
        return stru.pretty_print_table(tbl, name = "AMI PCT global flip mutation")

class apmi_remove_pct_mutation(mutation):
    def __init__(self, pct = 0.2):
        self.pct = pct

    def apply(self, model, data):

        # 1) Select a random rule to be removed (based on weight).
        feats = model.get_features()

        print("----------Removing---------")

        if len(feats) == 0:
            return model, {"changed": False}


        n = min(len(feats), math.ceil(len(feats) * self.pct))
        probs = [0.4*math.log2(len(f.all_conjoined_literals())) - 0.6*abs(w) for (f, w,_,_) in feats]
        probs = softmax_weights(probs)
        probs = None
        tbl = [("Weight", "len()", "log(len())", "P(sel)")]
        tbl = tbl + list(zip([w for (f,w,_,_) in feats],
                                    [len(f.all_conjoined_literals()) for (f,_,_,_) in feats],
                                    [math.log2(len(f.all_conjoined_literals())) for (f,_,_,_) in feats],
                                    probs))
        print(stru.pretty_print_table(tbl, top_bar=True, bot_bar=True))
        #print(tbl)

        removed_feats = list(np.random.choice(range(len(feats)), size = n, replace = False, p=probs ))
        removed_feats = [feats[i] for i in removed_feats]

        #print(model.to_string())

        for feat in removed_feats:
            model.remove_factor(feat)

        #print(model.to_string())

        print(f"removing {len(removed_feats)} feats!")
        for f in removed_feats:
            print(f)

        return model, {"changed":True, "removed":removed_feats}

    def applySDD(self, sdd, mgr, args):
        # sdd: ref()'d SDD
        # mgr: manager of SDD
        # args: "removed_ind" = indicator of removed rule.
        #print(args)

        if not args["changed"]:
            return sdd

        #print(args)

        indicators = [ind for (_,_,_,ind) in args["removed"]]
        return remove_indicators(sdd, mgr, indicators)

class apmi_add_pct_mutation(mutation):
    def __init__(self, gen, set_id="train", k = 3, pct = 0.15):
        self.gen = gen
        self.pct = pct
        self.k = k
        self.set_id = set_id

    def apply(self, model, data):
        # setup args
        args = {}

        print("--------------Adding-----------")

        # 1) sample an amount of features to add
        n = max(1, math.ceil(self.pct * len(model.get_features())))
        n = select_random_element(list(range(1,n+1)))

        # 2) generate all new features
        feats = [self.generate_feature(
            self.gen,
            self.k,
            data,
            model.count_manager) for i in range(n)]

        # 3) add all features
        feats = [model.add_factor(f,w) for (f,w) in feats]

        # 4) remove all features which could not be added to the model
        #    (doesn't affect model)
        feats = [e for e in feats if e is not None]

        # Nothing changed
        if len(feats) == 0:
            return model, {"changed":False}
        else:
            # all added feats
            print(f"Adding {len(feats)} new feats:")
            for f in feats:
                print(f)
            return model, {"changed":True, "added":feats}

    def applySDD(self, sdd, mgr, args):
        # args["changed"]: did the model chage at all?
        # args["added"]: all added rules


        if not args["changed"]:
            return sdd

        # Each new rule is a conjunction so can add all at the same time
        new_rules = conj([encoding for (_,_,encoding,_) in args["added"]])

        # new_rule referenced at top level only
        # referencing the new rule happens in .to_sdd()
        new_rules = new_rules.to_sdd(mgr)

        # new_sdd needs to be referenced again
        new_sdd = sdd & new_rules

        # Reference the new SDD
        new_sdd.ref()

        # dereference new_rule for auto gc and minimize.
        # The new rule is an intermediate step so should be derefed.
        new_rules.deref()

        return new_sdd

    def generate_feature(self, generator, k, dat, cmgr):
        candidates = [generator.random_feature_f() for i in range(k)]

        apmis = find_apmis_conjunctions(candidates, dat, self.set_id, cmgr)

        probs = [e/sum(apmis) for e in apmis]

        tbl = [("candidate", "APMI", "Probs")]
        tbl = tbl + list(zip(candidates, apmis, probs))

        print(stru.pretty_print_table(tbl))

        return (select_random_element(candidates, probs), generator.random_feature_w())

    def __str__(self):
        tbl = [("Prop", "Val"),
            ("PCT", self.pct),
            ("K", self.k),
            ("Measure", "AMI"),
            ("Generator", self.gen),
            ("Selection", "P(f) = AMI/ sum(AMI)"),
        ]
        return stru.pretty_print_table(tbl, name = "AMI PCT expand mutation")

def softmax_weights(weights):
    p = [math.exp(w) for w in weights]
    norm = sum(p)
    return [w/norm for w in p]

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

def select_random_element(elements, distr=None):
    index = np.random.choice(len(elements), replace=False, p = distr)
    return elements[index]

def find_amis_conjunctions(conjunctions, data, cmgr):
    return find_amis([f.list_of_factors for f in conjunctions], data, cmgr)

def find_amis(var_sets, data, cmgr):
    return [find_ami(var_set, data, cmgr) for var_set in var_sets]

def find_apmis_conjunctions(conjunctions, data, set_id, cmgr):
    return find_apmis([f.list_of_factors for f in conjunctions], data, set_id, cmgr)

def find_apmis(var_sets, data, set_id, cmgr):
    return [find_apmi(var_set, data, set_id, cmgr) for var_set in var_sets]

def pairwise_single_out(lst):
    return [(lst[i], lst[:i]+lst[i+1:]) for i in range(len(lst))]

def find_ami(lst, data, cmgr):
    pairs = pairwise_single_out(lst)
    pmis = [pmi(p[0], conj(p[1]), data, cmgr) for p in pairs]
    return sum(pmis)/len(pmis)

def find_apmi(var_set, data, set_id, cmgr):
    pairs = all_pairs(var_set)
    pmis = [pmi(p[0], p[1], data, set_id, cmgr) for p in pairs]
    return sum(pmis)/len(pmis)

def all_pairs(lst):
    if len(lst) == 1:
        return []

    head = lst[0]
    tail = lst[1:]

    pairs = [(head, r) for r in tail]
    rest_pairs = all_pairs(tail)

    return pairs + rest_pairs

def pmi(l, r, data, set_id, cmgr):
    Hl = entropy(l, data, set_id, cmgr)
    Hr = entropy(r, data, set_id, cmgr)
    Hlr= joint_entropy(l, r, data, set_id, cmgr)

    return Hl + Hr - Hlr

def ent(ps):
    if 0 in ps:
        return 0
    return -sum([p*math.log2(p) for p in ps])

def entropy(f, dat, set_id, cmgr):
    p = cmgr.count_factor(f, dat, set_id)
    np= cmgr.count_factor(neg(f), dat, set_id)
    return ent([p, np])

def joint_entropy(l, r, dat, set_id, cmgr):
    ps = [conj([l,r]), conj([l, neg(r)]), conj([neg(l), r]), conj([neg(l), neg(r)])]
    ps = [cmgr.count_factor(p, dat, set_id) for p in ps]
    return ent(ps)
