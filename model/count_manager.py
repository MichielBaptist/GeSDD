import numpy as np
from logic.conj import conj
from logic.lit import lit
from logic.neg import neg
import time

import utils.time_utils as tut

class count_manager:

    # ---------- API -----------------
    def __init__(self):
        self.cache = {}     # Cache empty first
        self.factor_cache = {}
        self.total_cache = {}

        self.cache_rate = 0
        self.access = 0
        self.t = tut.get_timer()

    def count_factors(self, factors, worlds, set_id):
        # factors: is a list [f] of factors from logic
        # worlds: ((c (w)), ..., (c (w)))
        # Returns a count of how often a factor is sat in worlds.

        # Delegate to self
        return [self.count_factor(f, worlds, set_id) for f in factors]

    def count_factor(self, factor, worlds, set_id):
        # factor: Is a factor from logic
        # worlds: ((), ..., ())

        # Do cache management ....

        t = self.t

        t.start()

        # --Debug stuff --
        self.access += 1

        t.t("Add 1")

        id = factor.unique_id() + set_id

        t.t("Calculate ID high level")

        b = id in self.cache

        t.t("Check cache")

        # Check cache otherwise count
        if b:

            # -- Debug stuff --
            self.cache_rate += 1

            t.t("Add 1 rate")

            res = self.cache[id]

            t.t("Get result from cache")

            return res
        else:
            #print(f"Counting uncached: {self.cache_rate/self.access}")
            if isinstance(factor, conj):
                counts = self.count_factor_conjunction(factor.list_of_factors, worlds, set_id)
            else:
                counts = sum([c*factor.evaluate(w) for (c, w) in worlds])/ sum([c for (c,_) in worlds])

            t.t("Calc counts last")

            self.cache[id] = counts

            t.t("Put result in cache")

            return counts

    def count_factor_conjunction_rec(self, conjunctors, worlds, set_id):
        if len(conjunctors) == 0:
            return sum([c for (c,_) in worlds]) / self.total_cache[set_id]

        new_worlds = self.get_sat_worlds_rec(conjunctors[0], worlds)

        return self.count_factor_conjunction_rec(conjunctors[1:], new_worlds, set_id)

    def count_factor_conjunction(self, conjunctors, worlds, set_id):
        t = self.t

        t.start()

        conjunctors_sat = [self.get_sat_worlds(c, worlds, set_id) for c in conjunctors]

        t.t("COUNT_CONJ--find sat worlds")

        all_sat = self.set_intersection(conjunctors_sat)

        t.t("COUNT_CONJ--Sat intersection")

        sm = sum([c for (c,_) in all_sat])

        t.t("COUNT_CONJ--Calc sum of counts")

        n = self.total_cache[set_id]

        t.t("COUNT_CONJ--Get total from cache")

        res = sm/n

        t.t("COUNT_CONJ--Calc result division")

        return res

    def set_intersection(self, lsts):
        return list(set.intersection(*lsts))

    def get_sat_worlds_rec(self, c, worlds):
        return [w for w in worlds if c.evaluate(w[1])]

    def get_sat_worlds(self, c, worlds, set_id):
        t = self.t

        t.start()

        id = c.unique_id() + set_id

        t.t("GET_SAT--Calc unique ID conjunctor")

        if not id in self.factor_cache:
            worlds = [w for w in worlds if c.evaluate(w[1])]
            t.t("GET_SAT--Calc sat worlds")
            worlds = set(worlds)
            t.t("GET_SAT--Cast to set")
            self.factor_cache[id] = worlds
            t.t("GET_SAT--Store in cache")
        else:
            worlds = self.factor_cache[id]
            t.t("GET_SAT--get from cache")

        return worlds

    def compress_data_set(self, data_set, set_id):
        # data_set: expects a tuple of tuplesv( ()... () )
        # returns a a list of tuples [(c, w)]
        #   Where c is the count and w is the world.

        n_vars = len(data_set[0])
        n_worlds = len(data_set)

        unique, count = np.unique(data_set, return_counts=True, axis=0)

        worlds = tuple([(c, tuple(w)) for (c,w) in zip(count, unique)])

        t = time.time()
        eval_fs = [lit(i) for i in range(1,n_vars+1)]
        eval_fs = eval_fs + [neg(lit(i)) for i in range (1, n_vars + 1)]
        for f in eval_fs:
            self.get_sat_worlds(f, worlds, set_id)

        print(f"Compressing dataset {set_id} took: {time.time() - t}")

        self.total_cache[set_id] = n_worlds

        return worlds

    def __repr__(self):
        lines = [
            "Standard count manager",
            f"--> Cache rate: {self.cache_rate / (1 if self.access == 0 else self.access)}"
        ]

        return "\n".join(lines)
    # -------- Internal ----------
