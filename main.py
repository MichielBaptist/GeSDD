from model.model import Model
from logic import *
from gen.gen import generator

import math
import gen.gen as g
import pysdd
import numpy as np

from datio import IO_manager

from graphviz import Source
from pysdd.sdd import SddManager
from ga.operations import *
from logic.conj import conj
from logic.disj import disj
from logic.lit  import lit
from logic.cons import cons
from logic.factor import factor
from logic.equiv import equiv

from matplotlib import pyplot as plt

from model.indicator_manager import indicator_manager

def __main__():
    n_vars = 10
    max_n_factors = 10
    bot = n_vars + 1
    top = bot + max_n_factors
    cross = simple_subset_cross()
    
    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    
    left = Model(n_vars, imgr, mgr, True)
    right = Model(n_vars, imgr, mgr, True)
    
    c1 = conj([lit(1), lit(2), lit(3)])
    d1 = disj([lit(-3), lit(-2), lit(5)])
    e1 = equiv([lit(1), lit(-6)])
    c2 = conj([lit(4), lit(2)])
    
    left.add_factor(c1)
    left.add_factor(e1)
    
    right.add_factor(c2)
    right.add_factor(d1)
    
    print(right.to_string())
    print(left.to_string())
    
    new_left, new_right = cross.cross(left, right)
    
def script2():

    data, n_worlds, n_vars = IO_manager.read_from_csv("data/movie/movie.train.data", ",", True)
    max_nb = 100
    n_sdds = 10
    amutation = add_mutation()
    rmutation = remove_mutation()    
    bot = n_vars + 1
    top = bot + max_nb
    
    mgr = SddManager(top)
    imgr = indicator_manager(range(bot, top))
    gen = generator(n_vars, imgr, fn_sparseness=0.005)
    
    models = gen.gen_n(n_sdds)
    
    print([model.nb_factors for model in models])
    
    [model.set_manager(mgr) for model in models]
    [model.update_sdd() for model in models]
    print([model.sdd_size() for model in models])
    
    sizes = list(map(lambda x : x.sdd_size(), models))
    ll = list(map(lambda x : x.LL(data), models))
    
    plt.plot(sizes, ll, 'bo')    
    
    [model.fit(data) for model in models]
    
    ll2 = list(map(lambda x : x.LL(data), models))
    
    plt.plot(sizes, ll2, 'ro')
    
    empty = Model(n_vars)
    empty.set_manager(mgr)
    empty.update_sdd()
    empty.dirty = True
    
    llb = empty.LL(data)
    empty.fit(data)    
    lla = empty.LL(data)
    
    plt.plot([0], [llb], "wo")
    plt.plot([0], [lla], "go")
    
    plt.show()
    
def script1():
    gen_n = 100
    vrs = 20
    max_nb_factors = 2000
    n_worlds = 1000
    bot = vrs + 1
    top = bot + max_nb_factors
    
    imgr = indicator_manager(range(bot, top))
    mgr = SddManager(top)
    gen = generator(vrs, imgr,operation_distr = [10, 10, 1])
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
