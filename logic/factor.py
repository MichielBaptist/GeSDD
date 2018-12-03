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
        
    def count(self, worlds):
        return sum(map(lambda x: self.evaluate(x), worlds))
    
    def __str__(self):
        return self.to_string()
    
    def is_literal(self):
        return False
    
    def __eq__(self, other):
        return False
        
    def __repr__(self):
        return self.__str__()
        
    pass
