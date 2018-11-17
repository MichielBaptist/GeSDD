from model.model import Model
from logic import *
from gen.gen import generator

import math
import gen.gen as g
import pysdd

from graphviz import Source
from pysdd.sdd import SddManager
from ga.operations import *

from matplotlib import pyplot as plt

def __main__():

    amutation = add_mutation()
    rmutation = remove_mutation()
    vrs = 3
    max_nb_f = 10
    
    left = Model(3)
    right = Model(3)
    
    mgr = SddManager(vrs + max_nb_f)
    
    left.set_manager(mgr)
    right.set_manager(mgr)
    
    amutation.mutate(right)
    amutation.mutate(right)
    print(right.to_string())
    
    Source(right.sdd.dot()).render(view=True)
    
    rmutation.mutate(right)
    rmutation.mutate(right)
    
    print(right.to_string())
    
    Source(right.sdd.dot()).render(view=True)
    
    
def script1():
    gen_n = 100
    vrs = 20
    max_nb_factors = 10
    n_worlds = 100
    
    mgr = SddManager(vrs + max_nb_factors)    
    gen = generator(vrs, operation_distr = [10, 10, 1], max_nb_factors = max_nb_factors)
    dat = g.random_data(vrs, n_worlds)
    
    models = [gen.gen() for i in range(gen_n)]
    [model.set_manager(mgr) for model in models]
    [model.update_sdd() for model in models]
    sizes = list(map(lambda x : x.sdd_size(), models))
    ll = list(map(lambda x : x.LL(dat), models))
    
    plt.plot(sizes, ll, 'bo')    
    
    [model.fit(dat) for model in models]
    
    ll2 = list(map(lambda x : x.LL(dat), models))
    
    plt.plot(sizes, ll2, 'ro')
    
    empty = Model(vrs)
    empty.set_manager(mgr)
    empty.update_sdd()
    empty.dirty = True
    
    llb = empty.LL(dat)
    empty.fit(dat)    
    lla = empty.LL(dat)
    
    plt.plot([0], [llb], "wo")
    plt.plot([0], [lla], "go")
    
    plt.show()
    
    
if __name__ == "__main__":
    __main__()
