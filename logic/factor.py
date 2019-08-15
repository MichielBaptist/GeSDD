class factor:
    """
        Class denoting a factor.
    """
    def __init__(self):
        self.id = None

    def to_string(self):
        return None

    def deref_all(self, lst):
        for sdd in lst:
            sdd.deref()

    def compile(self, manager):
        sdd = self.to_sdd(manager)
        sdd.ref()
        return sdd

    def to_sdd(self, manager):
        pass

    def evaluate(self, world):
        pass

    def is_literal(self):
        return False

    def __str__(self):
        return self.to_string()

    def __eq__(self, other):
        return False

    def __repr__(self):
        return self.__str__()

    def __init__(self):
        pass

    def unique_id(self):
        # If 2 different factors have the same string representation
        # then they represent the same logical function.
        res = self.id
        if res == None:
            res = self.to_string().replace(" ", "")
            self.id = res
        return res

    pass
