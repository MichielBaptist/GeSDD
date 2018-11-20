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
import ga.fitness

from logic.conj import conj
from logic.disj import disj
from logic.lit  import lit
from logic.cons import cons
from logic.factor import factor
from logic.equiv import equiv

from matplotlib import pyplot as plt

from model.indicator_manager import indicator_manager

def __main__():
    np.random.seed(40)
    data, struct, n_worlds, n_vars = IO_manager.read_from_names_folder("data/car")
    global_max = 200
    local_max = 200
    bot = n_vars + 1
    top = bot + global_max
    n_sdds = 20
    
    cross = simple_subset_cross()    
    fitness = ga.fitness.SimpleFitness(1, 0.01)
    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    gen = generator(n_vars, imgr, mgr, max_nb_factors = local_max)
    
    models = gen.gen_n(n_sdds)
    
    print([model.nb_factors for model in models])
    
    [model.set_manager(mgr) for model in models]
    [model.update_sdd() for model in models]
    print([model.sdd_size() for model in models])
    
    sizes = list(map(lambda x : math.log(x.sdd_size()), models))
    ll = list(map(lambda x : x.LL(data), models))
    
    plt.plot(sizes, ll, 'bo')    
    
    [model.fit(data) for model in models]
    
    ll2 = list(map(lambda x : x.LL(data), models))
    
    plt.plot(sizes, ll2, 'ro')
    
    empty = Model(n_vars)
    empty.set_manager(mgr)
    empty.update_sdd()
    empty.dirty = True
    
    custom = Model(n_vars)
    custom.set_manager(mgr)
    
    not_high_buying_cost = lit(-struct.variable_of(0, "high"))
    not_high_maint_cost = lit(-struct.variable_of(1, "high"))
    doors_5 = lit(struct.variable_of(2, "5more"))
    
    pers2 = lit(struct.variable_of(3, "2"))
    pers4 = lit(struct.variable_of(3, "4"))
    morepers = lit(struct.variable_of(3, "more"))
    npers2 = lit(-struct.variable_of(3, "2"))
    npers4 = lit(-struct.variable_of(3, "4"))
    nmorepers = lit(-struct.variable_of(3, "more"))
    
    lug_buut = lit(struct.variable_of(4, "big"))
    not_high_safety = lit(-struct.variable_of(5, "high"))
    not_low_safety = lit(-struct.variable_of(5, "low"))
    
    very_good_car = lit(struct.variable_of(6, "vgood"))
    good_car = lit(struct.variable_of(6, "good"))    
    unacc = lit(struct.variable_of(6, "unacc"))
    
    good_or_vgood = disj([very_good_car, good_car]) 
       
    f1 = disj([good_or_vgood, not_high_safety] )
    f2 = disj([not_low_safety, unacc])
    f3 = disj([not_high_buying_cost, not_high_maint_cost, unacc])
    f4 = disj([conj([pers2, npers4, nmorepers]),
               conj([npers2, pers4, nmorepers]),
               conj([npers2, npers4, morepers])])
    f5 = disj([morepers, doors_5])
    
    custom.add_factor(f1)
    custom.add_factor(f2)
    custom.add_factor(f3)
    custom.add_factor(f4)
    custom.add_factor(f5)
    
    custom.update_sdd()
    custom.dirty = True
    
    llbc = custom.LL(data)
    custom.fit(data)
    llac = custom.LL(data)
    
    llb = empty.LL(data)
    empty.fit(data)    
    lla = empty.LL(data)
    
    plt.plot([math.log(custom.sdd_size())], [llbc], "yo")
    plt.plot([math.log(custom.sdd_size())], [llac], "go")
    plt.plot([0], [llb], "yo")
    plt.plot([0], [lla], "go")
    
    mxi = ll2.index(max(ll2))
    
    print(models[mxi].to_string())
    print(custom.to_string())
    
    plt.show()
    
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
