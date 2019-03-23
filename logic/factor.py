class factor:
    """
        Class denoting a factor.
    """
    def to_string(self):
        return None

    def to_sdd(self, manager):
        pass

    def ref(self):
        pass

    def deref(self):
        pass

    def compile(self, manager):
        return self.to_sdd(manager)

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
        return self.to_string().replace(" ", "")

    pass
