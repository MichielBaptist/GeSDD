from model.model import Model
from logic import *
from gen.gen import generator

import math
import gen.gen as g
import pysdd
import numpy as np
import sys
import os

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
        [model.fit(train, 1e-10, acc) for model in sdds] # Fit all models with accuracy

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




    plt.subplot(2,1,1)
    plt.plot(accuracies, avll_train, label="Training Log-Liklihood")
    plt.plot(accuracies, avll_test, label="Validation Log-Liklihood")

    plt.xlabel("Accuraccy")
    plt.ylabel("Log liklihood")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(accuracies, time_, label="Time taken")

    plt.xlabel("Accuraccy")
    plt.ylabel("Time taken (s)")
    plt.legend()

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

    #data, n_worlds, n_vars = IO_manager.read_from_csv_num("data/nltcs/nltcs.train.wdata", ",")
    #quit()
    n_worlds = 1200
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

    # 3) split into train and validation
    train, valid = train_test_split(data, test_size = 0.15)
    train = tuple(train)
    valid = tuple(valid)
    
    custom.partition()

    print(train)
    print(valid)

    print(custom.to_string())
    print(custom.count(custom.get_f(), train))
    #quit()


    #train, valid = train_test_split(data, test_size = 0.15)

    # --- Get the base LL (empty model) ---
    empty = Model(n_vars, imgr, mgr)
    empty.update_sdd()
    empty.fit(train)
    base = empty.LL(train)
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
    params['train'] = train
    params['valid'] = valid
    params['mutate_probability'] = 0.8
    params['n_gens'] = 100
    params['n_select'] = 10


    # Add base fitness and ll to the log
    params['logbook'].post(0, "base_ll_tr", custom.LL(train))
    params['logbook'].post(0, "base_fit_tr", params['fitness'].of(custom, train))
    params['logbook'].post(0, "base_ll_va", custom.LL(valid))
    params['logbook'].post(0, "base_fit_va", params['fitness'].of(custom, valid))

    pop, fits = algorithm.run(params)

    aggregators = [
        LL_FIT_MAX_AVG_MIN,
        TIMES
    ]
    svr = saver("run")
    svr.save_run(params, logbook, aggregators)


def TIMES(log, save_path):
    t_fit = np.mean(log.get_prop("time: fitting"))
    t_mut = np.mean(log.get_prop("time: mutation"))

    print(t_fit, t_mut)

def LL_FIT_MAX_AVG_MIN(log, save_path):

    max_fit_t = [max(it) for it in log.get_prop("fit_t")]
    avg_fit_t = [np.mean(it) for it in log.get_prop("fit_t")]
    min_fit_t = [min(it) for it in log.get_prop("fit_t")]

    max_fit_v = [max(it) for it in log.get_prop("fit_v")]
    avg_fit_v = [np.mean(it) for it in log.get_prop("fit_v")]
    min_fit_v = [min(it) for it in log.get_prop("fit_v")]

    min_ll_t = [min(it) for it in log.get_prop("ll_t")]
    max_ll_t = [max(it) for it in log.get_prop("ll_t")]
    avg_ll_t = [np.mean(it) for it in log.get_prop("ll_t")]

    min_ll_v = [min(it) for it in log.get_prop("ll_v")]
    max_ll_v = [max(it) for it in log.get_prop("ll_v")]
    avg_ll_v = [np.mean(it) for it in log.get_prop("ll_v")]

    goal_ll_t = log.get_prop("base_ll_tr")[0]
    goal_fit_t = log.get_prop("base_fit_tr")[0]
    goal_ll_v = log.get_prop("base_ll_va")[0]
    goal_fit_v = log.get_prop("base_fit_va")[0]

    # Create time plot
    times = log.get_prop("time")
    times_fitting = log.get_prop("time: fitting")
    times_cross = log.get_prop("time: crossing")
    times_mutation = log.get_prop("time: mutation")

    plt.plot(range(len(times)), times, label="Times per iteration")
    plt.plot(range(len(times)), times_fitting, label="Times per iteration (fitting)")
    plt.plot(range(len(times)), times_cross, label="Times per iteration (cross)")
    plt.plot(range(len(times)), times_mutation, label="Times per iteration (mutation)")
    plt.xlabel("Generation")
    plt.ylabel("Time (s)")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "Times"))

    plt.clf()
    plt.cla()
    plt.close()

    # Create validation ll plot

    plt.plot(range(1, len(min_ll_t) + 1), min_ll_v, label="min LL (V)")
    plt.plot(range(1, len(avg_ll_t) + 1), avg_ll_v, label="pop LL (V)")
    plt.plot(range(1, len(max_ll_t) + 1), max_ll_v, label="max LL (V)")
    plt.plot(range(1, len(max_ll_t) + 1), [goal_ll_v for i in range(len(max_ll_t))], label="Goal (V)")
    plt.xlabel("Generation")
    plt.ylabel("Log Liklihood")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, 'validation_ll_plot.pdf'))

    plt.clf()
    plt.cla()
    plt.close()

    # Create training ll plot
    plt.plot(range(1, len(min_ll_t) + 1), min_ll_t, label="min LL (T)")
    plt.plot(range(1, len(avg_ll_t) + 1), avg_ll_t, label="pop LL (T)")
    plt.plot(range(1, len(max_ll_t) + 1), max_ll_t, label="max LL (T)")
    plt.plot(range(1, len(max_ll_t) + 1), [goal_ll_t for i in range(len(max_ll_t))], label="Goal (T)")
    plt.xlabel("Generation")
    plt.ylabel("Log Liklihood")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "training_ll_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()

    # Create validation fit plot


    # Create training fit plot
    plt.plot(range(1, len(avg_fit_t) + 1), avg_fit_t, label = "populaion fitness (T)")
    plt.plot(range(1, len(max_fit_t) + 1), max_fit_t, label = "maximum fitness (T)")
    plt.plot(range(1, len(min_fit_t) + 1), min_fit_t, label = "minimum fitness (T)")
    plt.plot(range(1, len(max_ll_t) + 1), [goal_fit_t for i in range(len(max_ll_t))], label="Goal (T)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "training_fit_plot.pdf"))
    plt.clf()
    plt.cla()
    plt.close()

    # Create validation fit plot
    plt.plot(range(1, len(avg_fit_t) + 1), avg_fit_v, label = "populaion fitness (V)")
    plt.plot(range(1, len(max_fit_t) + 1), max_fit_v, label = "maximum fitness (V)")
    plt.plot(range(1, len(min_fit_t) + 1), min_fit_v, label = "minimum fitness (V)")
    plt.plot(range(1, len(max_ll_t) + 1), [goal_fit_v for i in range(len(max_ll_t))], label="Goal (V)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "validation_fit_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()

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
