class factor:
    """
        Class denoting a factor.
    """
    def to_string(self):
        return None

    def to_sdd(self, manager):
        pass

    def compile(self, manager):
        if self.get_sdd() == None:
            self.sdd = self.to_sdd(manager)
        return self.get_sdd()

    def get_sdd(self):
        return self.sdd

    def evaluate(self, world):
        pass

    def has_count(self, name):
        return name in self.counts

    def get_count(self, name):
        return self.counts[name]

    def set_count(self, name, count):
        self.counts[name] = count

    def count(self, worlds):
        # Very primitive caching of the counts, but works for now
        if self.has_count(worlds):
            return self.get_count(worlds)
        else:
            #Count and cache this count
            self.set_count(worlds, sum(map(lambda x: self.evaluate(x), worlds)))
            return self.count(worlds)

    def __str__(self):
        return self.to_string()

    def is_literal(self):
        return False

    def __eq__(self, other):
        return False

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        self.counts = {}

    pass
