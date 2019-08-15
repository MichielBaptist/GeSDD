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
from logic.neg import neg

import utils.string_utils as stru

class generator:
    def __init__(self):
        pass

    def gen(self):
        pass

    def gen_n(self, n):
        return [self.gen() for i in range(n)]

    def random_feature(self):
        return (self.random_feature_f(), self.random_feature_w())

    def random_feature_w(self):
        pass

    def random_feature_f(self):
        pass

    def __str__(self):
        return "generator super class"

class random_generator(generator):
    def __init__(self,
                domain_size,
                mgr_top,
                cmgr,
                nb_threshold = 5,
                max_feat_len = 0.6):
        self.domain_size = domain_size
        self.nb_threshold= nb_threshold
        self.mgr_top = mgr_top
        self.cmgr = cmgr
        self.max_feat_len = max_feat_len
        pass

    def gen(self):
        # 1) Samples the amount of factors in the set.
        nb_factors = np.random.choice(range(1, self.nb_threshold + 1))

        #nb_factors = self.nb_threshold

        # 2) create the factors
        factors = [self.random_feature() for i in range(nb_factors)]

        # 3) create the model and add features
        mdl = Model(domain_size = self.domain_size,
                    manager = SddManager(self.mgr_top, True),
                    #indicator_manager = self.indicator_manager,
                    count_manager = self.cmgr,
        )
        for (f, w) in factors:
            mdl.add_factor(f, w)

        # 4) initial compilation
        mdl.compile()

        return mdl

    def random_feature_f(self):
        #1) random feat_n
        feat_n = 1 + np.random.geometric(0.6)
        feat_n = min(feat_n, int(self.domain_size))

        #2) sample subset
        subset = np.random.choice(range(1, self.domain_size + 1), replace = False, size = feat_n)
        subset = sorted(subset)

        #3) to lits + negate randomly
        lits = [lit(l) for l in subset]
        list = [self.negate_pr(l) for l in lits]

        return conj(lits)

    def negate_pr(self, l):
        if random.random() < 0.5:
            return neg(l)
        else:
            return l

    def random_feature_w(self):
        return 1

class data_seeded_generator(generator):
    def __init__(self,
                 seed_data,
                 indicator_manager,
                 count_manager,
                 mgr_top,
                 set_threshold = 5):
        self.seed_data = seed_data
        self.indicator_manager = indicator_manager
        self.count_manager = count_manager
        self.domain_size = self.extract_ds(seed_data)
        self.mgr_top = mgr_top

        self.set_threshold = set_threshold

        fs, fp = self.parse_feature_set(seed_data)

        self.feature_set = fs
        self.feature_probs = fp

    def gen(self):
        # 1) Samples the amount of factors in the set.
        nb_factors = np.random.choice(range(1, self.set_threshold + 1))

        # 2) create the factors
        factors = [self.random_feature() for i in range(nb_factors)]

        # 3) create the model and add features
        mdl = Model(domain_size = self.domain_size,
                    manager = SddManager(self.mgr_top, True),
                    #indicator_manager = self.indicator_manager,
                    count_manager = self.count_manager,
        )
        for (f, w) in factors:
            mdl.add_factor(f, w)

        # 4) initial compilation
        mdl.compile()

        return mdl

    def random_feature_f(self):
        # 1) sample random feature
        feat = np.random.choice(a = self.feature_set, p = self.feature_probs)

        # 2) drop random amount of lits
        feat = self.drop_rnd(feat, min = 2)

        # 3) thresholding?

        return conj([lit(lt) for lt in feat])

    def random_feature_w(self):
        return 1

    def drop_rnd(self, lst, min = 0):
        # Sample a random length

        if len(lst) <= min:
            return lst

        rnd_drp_len = np.random.choice(range(1, len(lst)-1))
        rnd_drp_ind = np.random.choice(range(len(lst)), rnd_drp_len, replace=False)

        return [el for ind, el in enumerate(lst) if ind not in rnd_drp_ind]

    def parse_feature_set(self, seed_data):
        # seed_data: [(c, w)]

        fs = [ self.to_feature_ind(w) for (c, w) in seed_data if self.correct_fs(w)]
        fp = [ 1/(len(fs)) for i in range(len(fs))] # Uniform

        return fs, fp

    def correct_fs(self, world):
        return sum(world) >= 2

    def to_feature_ind(self, world):
        # world: (True, False, False, ..., True)
        return [ind + 1 for ind, elm in enumerate(world) if elm]

    def extract_ds(self, dat):
        # dat: [(c, w)]
        return len(dat[0][1])
