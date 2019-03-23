import numpy as np

class count_manager:

    # ---------- API -----------------
    def __init__(self):
        self.cache = {}     # Cache empty first

        self.cache_rate = 0
        self.access = 0

    def count_factors(self, factors, worlds):
        # factors: is a list [f] of factors from logic
        # worlds: ((c (w)), ..., (c (w)))
        # Returns a count of how often a factor is sat in worlds.

        # Delegate to self
        return [self.count_factor(f, worlds) for f in factors]

    def count_factor(self, factor, worlds):
        # factor: Is a factor from logic
        # worlds: ((), ..., ())

        # Do cache management ....

        # --Debug stuff --
        self.access += 1

        # Check cache otherwise count
        if (factor.unique_id(), worlds) in self.cache:

            # -- Debug stuff --
            self.cache_rate += 1

            return self.cache[(factor.unique_id(), worlds)]
        else:
            counts = sum([c*factor.evaluate(w) for (c, w) in worlds])/ sum([c for (c,_) in worlds])
            self.cache[(factor.unique_id(), worlds)] = counts
            return self.count_factor(factor, worlds)

    def compress_data_set(self, data_set):
        # data_set: expects a tuple of tuplesv( ()... () )
        # returns a a list of tuples [(c, w)]
        #   Where c is the count and w is the world.

        unique, count = np.unique(data_set, return_counts=True, axis=0)

        return tuple([(c, tuple(w)) for (c,w) in zip(count, unique)])

    def __repr__(self):
        lines = [
            "Standard count manager",
            f"--> Cache rate: {self.cache_rate / (1 if self.access == 0 else self.access)}"
        ]

        return "\n".join(lines)
    # -------- Internal ----------
