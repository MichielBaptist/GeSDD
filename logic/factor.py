class factor:
    """
        Class denoting a factor.
    """
    def to_string(self):
        return None
    def to_sdd(self, manager):
        pass
    def evaluate(self, world):
        pass
    def count(self, worlds):
        return sum(map(lambda x: self.evaluate(x), worlds))
    
    def __str__(self):
        return self.to_string()
    
    def is_literal(self):
        return False
    
    pass
