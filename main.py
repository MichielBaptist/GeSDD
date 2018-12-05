from model.model import Model
from logic import *
from gen.gen import generator

import math
import gen.gen as g
import pysdd
import numpy as np
import sys

from Saver import saver

import data_gen

from datio import IO_manager

from graphviz import Source
from pysdd.sdd import SddManager

import ga.operations as gao
from ga.operations import *
import ga.fitness
import ga.alg as algorithm
from ga.alg import logbook

from logic.conj import conj
from logic.disj import disj
from logic.lit  import lit
from logic.cons import cons
from logic.factor import factor
from logic.equiv import equiv

from matplotlib import pyplot as plt

from model.indicator_manager import indicator_manager

from sklearn.model_selection import train_test_split

def data_gen_script():
    
    n_vars = 10
    global_max = 30
    local_max = 30
    bot = n_vars + 1
    top = bot + global_max
    
    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    
    
    model = Model(n_vars, imgr, mgr)
    model.dynamic_update = True
    model.update_sdd()
    
    model = custom_model(model)
    
    # Gen data from model
    dsizes = [int(math.pow(10,i)) for i in range(4,5)]
    sets = [data_gen.gen(model, size) for size in dsizes]
    
    '''
    # Define sdds
    sdds = [model]
    
    # Define accurcaies    
    accuracies = [math.pow(10,-i/100) for i in range(0, 400, 5)]
    print(accuracies)
    
    for dset in sets:
        train, test = train_test_split(dset, test_size = 0.1)
        avll_train = []
        avll_test = []
        time_ = []
        for acc in accuracies:        
            [model.set_factor_weights([0 for i in range(len(model.factors))]) for model in sdds]
            
            start = time.time()
            [model.fit(train, acc) for model in sdds] # Fit all models with accuracy
            
            end = time.time()
            
            lls_train = [model.LL(train) for model in sdds]
            lls_test = [model.LL(test) for model in sdds]
            
            avg = np.mean(lls_train)
            avg_test = np.mean(lls_test)
            
            avll_train.append(avg)
            avll_test.append(avg_test)
            time_.append(end-start)
            
        
        plt.semilogx(accuracies, avll_train)
        plt.semilogx(accuracies, avll_test)
        plt.semilogx(accuracies, time_)
        
    print(model.to_string())
    plt.show()
    
    '''
    
def custom_model2(model):
    f1 = disj([lit(-5), lit(-1), lit(2) ])
    f2 = disj([lit(-5), lit(-2), lit(1) ])
    f3 = disj([lit(-1),lit(3)])
    f4 = disj([lit(-2),lit(4)])
    f5 = disj([lit(-1),lit(-2), lit(5)])
    #f6 = disj([])
    
    model.add_factor(f1, 2)
    model.add_factor(f2, 2)    
    model.add_factor(f3, 3)
    model.add_factor(f4, 3)
    model.add_factor(f5, 1)
    return model
    
def custom_model(model):
    # Custom model here
    
    f1 = conj([lit(1), lit(2), lit(-3)])
    f2 = disj([lit(-4), lit(6)])
    f3 = disj([lit(5), lit(-7), lit(8), lit(-9)])
    f4 = conj([lit(-4), lit(10)])
    f5 = equiv([lit(9), lit(6)])
    
    model.add_factor(f1, 2)
    model.add_factor(f2, 3)
    model.add_factor(f3, 1.5)
    model.add_factor(f4, 0.9)
    model.add_factor(f5, 2)
    
    return model
    
def script4():
    np.random.seed()
    
    data, struct, n_worlds, n_vars = IO_manager.read_from_names_folder("data/car")
    train, test = train_test_split(data, test_size = 0.1)
    global_max = n_vars + 400
    local_max = 20
    bot = n_vars + 1
    top = bot + global_max
    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    gen = generator(n_vars, imgr, mgr, max_nb_factors = local_max)
    
    sdds = gen.gen_n(1)
        
    avll_train = []
    avll_test = []
    time_ = []
    
    # Define accurcaies
    
    
    #accuracies = [math.exp(-i/10) for i in range(0, 70, 1)]
    accuracies = [i for i in range(100)]
    print(accuracies)
    print(len(train))
    print(len(test))
    
    # -----------------
    
    
    for acc in accuracies:
        start = time.time()
        [model.fit(train, 0, acc) for model in sdds] # Fit all models with accuracy
        
        end = time.time()
        
        lls_train = [model.LL(train) for model in sdds]
        lls_test = [model.LL(test) for model in sdds]
        
        avg = np.mean(lls_train)
        avg_test = np.mean(lls_test)
        
        avll_train.append(avg)
        avll_test.append(avg_test)
        time_.append(end-start)
        
        print(acc)
        
        [model.set_factor_weights([0 for i in range(len(model.factors))]) for model in sdds]
        
    plt.plot(accuracies, avll_train)
    plt.plot(accuracies, avll_test)
    plt.plot(accuracies, time_)
    plt.show()
         
def frange(start, end, step):
    if start < end:
        i = start
        while i < end:
            yield i
            i+= step
            
    else:
        i = start
        while end < i:
            yield i
            i -= step
    
def __main__():
    # Generate a random seed
    seed = random.randrange(2^32 - 1)
    print(seed)
    rng = random.Random(seed)
    print("Seed was:", seed)

    # Set the seed for reproducing results
    np.random.seed(seed)
    random.seed(seed)
    
    #data, struct, n_worlds, n_vars = IO_manager.read_from_names_folder("data/car")
    n_worlds = 1000
    n_vars = 5
    global_max = 400
    local_max = 200
    bot = n_vars + 1
    top = bot + global_max
    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    gen = generator(n_vars, imgr, mgr, max_nb_factors = local_max)
    
    # --- Generate the data ---------------
    # 1) custom SDD
    custom = Model(n_vars, imgr, mgr)
    custom.dynamic_update = True
    goal = custom_model2(custom)
    
    # 2) sample the SDD
    data = data_gen.gen(custom, n_worlds)
    
    custom.partition()
    
    print(custom.to_string())
    print(custom.count(custom.get_f(), data))
    #quit()
    
    # --- Get the base LL (empty model) ---
    empty = Model(n_vars, imgr, mgr)
    empty.update_sdd()
    empty.fit(data)
    base = empty.LL(data)
    print(base)
    # -------------------------------------
    
    params = {}
    params['seed'] = seed
    params['selection'] = gao.Weighted_selection(1)
    params['paring'] = gao.pairing()
    params['cross'] = gao.rule_swapping_cross()
    params['mutation'] = gao.script_mutation([
                                             gao.threshold_mutation(0.1),
                                             gao.multi_mutation([
                                                    gao.add_mutation(),
                                                    gao.weighted_remove_mutation()
                                                ]),
                                             ])
                                             
    params['fitness'] = ga.fitness.fitness2(0.5, base)
    params['generator'] = gen
    params['logbook'] = logbook()    
    params['pop_size'] = 20
    params['data'] = data
    params['mutate_probability'] = 0.8
    params['n_gens'] = 50
    params['n_select'] = 10
    
    pop, fits = algorithm.run(params)
    
    svr = saver("run")
    svr.save_run(params, logbook)
    
    log = params['logbook']
    
    max_fit = [max(it) for it in log.get_prop("fit")]
    max_ll = [max(it) for it in log.get_prop("ll")]
    avg_fit = log.get_prop("fit")
    avg_fit = [np.mean(it) for it in avg_fit]
    
    avg_ll = log.get_prop("ll")
    avg_ll = [np.mean(it) for it in avg_ll]
    
    goal_ll = custom.LL(data)
    goal_fit = params['fitness'].of(custom, data)
    
    plt.plot(range(1, len(avg_ll) + 1), avg_ll)    
    plt.plot(range(1, len(avg_fit) + 1), avg_fit) 
    plt.plot(range(1, len(max_fit) + 1), max_fit) 
    plt.plot(range(1, len(max_ll) + 1), max_ll)
    plt.plot(range(1, len(max_ll) + 1), [goal_ll for i in range(len(max_ll))])
    plt.plot(range(1, len(max_ll) + 1), [goal_fit for i in range(len(max_ll))])
    plt.show()
    
    print(avg_fit)
    print(max_ll)
    
def script3():
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
