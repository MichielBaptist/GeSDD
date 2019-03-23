from model.model import Model
from logic import *
from gen.gen import recursive_generator

import math
import gen.gen as g
import pysdd
import numpy as np
import sys
import os

from Saver import saver

import data_gen

from graphviz import Source

from datio import IO_manager
from pysdd.sdd import SddManager
from model.indicator_manager import indicator_manager
from model.count_manager import count_manager

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
from logic.neg import neg
from logic.impl import impl

from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split

def data_gen_script():

    n_vars = 5
    n_worlds = 120000

    mgr = SddManager(20)


    model = Model(n_vars, None, mgr)
    model = custom_model4(model)
    model.dynamic_update = True
    model.update_sdd()
    model.partition()

    print(f"SDD Size: {model.sdd_size()}")
    print(model.to_string())

    worlds = data_gen.gen(model, n_worlds)

    train, valid = train_test_split(worlds, test_size = 0.15)

    print(np.shape(train))
    print(np.shape(valid))

    print(f"SDD Size: {model.sdd_size()}")

    model.sdd.ref()
    mgr.minimize()
    model.sdd.deref()


    print(f"SDD Size: {model.sdd_size()}")

    print(f"Train LL: {model.LL(train)}")
    print(f"Valid LL: {model.LL(valid)}")

    #data_gen.save(f"data/synthetic/synth_train_{n_vars}_2.dat", train, ',')
    #data_gen.save(f"data/synthetic/synth_valid_{n_vars}_2.dat", valid, ',')

    quit()




def custom_model5(model):
    a = lit(1)
    _a = neg(a)
    b = lit(2)
    _b = neg(b)
    c = lit(3)
    _c = neg(c)
    d = lit(4)
    _d = neg(d)
    e = lit(5)
    _e = neg(e)
    f = lit(6)
    _f = neg(f)
    g = lit(7)
    _g = neg(g)
    h = lit(8)
    _h = neg(h)

    f1 = impl(conj([a, b, _c]), conj([_d, e]))
    f2 = conj([a, _b, _c, _h])
    f3 = disj([_a, f, _g])
    f4 = impl(conj([h, g, d]), conj([_e, d]))
    f5 = impl(conj([_f, d]), _h)
    f6 = impl(conj([_g, _d]), conj([_h, a]))

    model.add_factor(f1, 3)
    model.add_factor(f2, 1)
    model.add_factor(f3, -2)
    model.add_factor(f4, 2)
    model.add_factor(f5, 1.5)
    model.add_factor(f6, -1)

    return model

def custom_model4(model):
    f1 = disj([lit(-5), lit(-1), lit(2) ])
    f1 = impl(conj([lit(5), lit(1)]), lit(2))

    f2 = disj([lit(-5), lit(-2), lit(1) ])
    f2 = impl(conj([lit(5), lit(2)]), lit(1))

    f3 = disj([lit(-1),lit(3)])
    f3 = impl(lit(1), lit(3))

    f4 = disj([lit(-2),lit(4)])
    f4 = impl(lit(2), lit(4))

    f5 = disj([lit(-1),lit(-2), lit(5)])
    f5 = impl(conj([lit(1), lit(2)]), lit(5))

    # Random rule
    f6 = impl(conj([neg(lit(1)), neg(lit(4))]), conj([lit(5), lit(2)]))

    #f6 = disj([])

    model.add_factor(f1, 2)
    model.add_factor(f2, 2)
    model.add_factor(f3, 3)
    model.add_factor(f4, 3)
    model.add_factor(f5, 1)
    model.add_factor(f6, 2)

    return model

def custom_model3(model):
    Atakes1 = lit(1)
    Btakes1 = lit(2)
    Atakes2 = lit(3)
    Btakes2 = lit(4)
    ABgroup1 = lit(5)
    ABfriends = lit(6)

    f1 = impl(ABfriends, conj([Atakes1, Btakes1]))
    f2 = impl(ABfriends, conj([Atakes2, Btakes2]))
    f3 = impl(conj([Atakes1, neg(Btakes1)]), neg(ABgroup1))
    f4 = impl(conj([neg(Atakes1), Btakes1]), neg(ABgroup1))
    f5 = impl(ABgroup1, ABfriends)

    model.add_factor(f1, 2)
    model.add_factor(f2, 2)
    model.add_factor(f3, 5)
    model.add_factor(f4, 5)
    model.add_factor(f5, 3)

    return model

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
    imgr = indicator_manager(range(bot, top))
    cmgr = count_manager()
    gen = recursive_generator(n_vars, imgr, mgr, max_nb_factors = local_max)

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

    data_files = ("data/synthetic/synth_train_5_2.dat", "data/synthetic/synth_valid_5_2.dat")
    #data_files = None

    # Generate a random seed
    seed = random.randrange(2**20)
    print("Seed was:", seed)

    # Set the seed for reproducing results
    np.random.seed(seed)
    random.seed(seed)

    if data_files != None:
        print("Using data files!")
        train_path, valid_path = data_files
        train, n_worlds_train, n_vars = IO_manager.read_from_csv(train_path, ',')
        valid, n_worlds_valid, n_vars = IO_manager.read_from_csv(valid_path, ',')
        n_worlds = n_worlds_train + n_worlds_valid
        test_train_ratio = n_worlds_valid / n_worlds
    else:
        n_worlds = 1000000
        n_vars = 5
        print("Generating own data using:")
        print(stru.pretty_print_table([
                        ("Worlds:", str(n_worlds)),
                        ("Variables:", str(n_vars))]))

    global_max = 400
    local_max = 200
    bot = n_vars + 1
    top = bot + global_max
    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    cmgr = count_manager()
    gen = recursive_generator(n_vars, imgr, mgr, cmgr, max_nb_factors = local_max)


    if data_files == None:
        test_train_ratio = 0.15
        custom = Model(n_vars, imgr, mgr, cmgr)
        custom.dynamic_update = True
        goal = custom_model4(custom)
        custom.update_sdd()

        custom.sdd.ref()

        print(f"Dead before minimization: {mgr.dead_size()}")
        print(f"Live before minimization: {mgr.live_size()}")
        print(f"SDD size before minimization: {custom.sdd_size()}")

        mgr.minimize()

        print(f"Dead after minimization: {mgr.dead_size()}")
        print(f"Live after minimization: {mgr.live_size()}")
        print(f"SDD size after minimization: {custom.sdd_size()}")

        custom.sdd.deref()

        data = data_gen.gen(custom, n_worlds)
        train, valid = train_test_split(data, test_size = 0.15)

    else:
        custom = None

    train = cmgr.compress_data_set(train)
    valid = cmgr.compress_data_set(valid)

    params = {}
    params['seed'] = seed
    params['selection'] = gao.Weighted_selection(1)
    params['paring'] = gao.pairing()
    params['cross'] = gao.rule_swapping_cross()
    params['mutation'] = gao.script_mutation([
                                             gao.threshold_mutation(0.1),
                                             gao.multi_mutation([
                                                    gao.add_mutation(),
                                                    gao.weighted_remove_mutation(),
                                                    gao.rule_expansion_mutation(),
                                                    gao.rule_shrinking_mutation(),
                                                    gao.sign_flipping_mutation()
                                                ]),
                                             ])

    params['fitness'] = ga.fitness.fitness3(0.0002)
    params['generator'] = gen
    params['logbook'] = logbook()
    params['pop_size'] = 100
    params['train'] = train
    params['valid'] = valid
    params['mutate_probability'] = 0.8
    params['n_gens'] = 150
    params['n_select'] = 50
    params['manager'] = mgr
    params['indicator_manager'] = imgr
    params['n_worlds'] = n_worlds
    params['n_vars'] = n_vars
    params['test_train_ratio'] = test_train_ratio
    if custom != None:
        params['custom_model'] = custom.to_string()
        params['logbook'].post(0, "base_ll_tr", custom.LL(train))
        params['logbook'].post(0, "base_fit_tr", params['fitness'].of(custom, train))
        params['logbook'].post(0, "base_ll_va", custom.LL(valid))
        params['logbook'].post(0, "base_fit_va", params['fitness'].of(custom, valid))
        params['logbook'].post(0, "base_size", custom.sdd_size())
    if data_files != None:
        train_path, valid_path = data_files
        params['train_file'] = train_path
        params['valid_file'] = valid_path

    pop, fits = algorithm.run(params)

    params['manager'] = None

    # ZERO MODEL STUFF
    zero_model = Model(n_vars, imgr, mgr, cmgr)
    zero_model.update_sdd()
    zero_model.fit(train)
    zero_model.fit(train)

    zero_t_ll = zero_model.LL(train)
    zero_v_ll = zero_model.LL(valid)
    zero_t_fit = params['fitness'].of(zero_model, train)
    zero_v_fit = params['fitness'].of(zero_model, valid)

    params['logbook'].post(0, "zero_t_ll", zero_t_ll)
    params['logbook'].post(0, "zero_v_ll", zero_v_ll)
    params['logbook'].post(0, "zero_t_fit", zero_t_fit)
    params['logbook'].post(0, "zero_v_fit", zero_v_fit)
    # ZERO MODEL stuff

    aggregators = [
        FITNESS_TRAIN,
        FITNESS_VALID,
        LL_TRAIN,
        LL_VALID,
        TIMES,
        SIZES,
        BEST_AND_SIZE,
        BEST_IND,
        NB_FACTORS,
        LIVE_DEAD_SIZE,
        INDICATOR_PROFILE,
        BEST_MODEL,
        MODEL_EFFICIENY,
    ]
    svr = saver("run")
    svr.save_run(params, logbook, aggregators)

def MODEL_EFFICIENY(log, save_path):
    bests = log.get_prop("best_model")
    zero_v_ll = log.get_point(0, "zero_v_ll")

    efficiencies = [ m.sdd_size() / (vl - zero_v_ll) for (m, _, _, vl,_) in bests]
    v_lls = [vl for (_,_,_,vl,_) in bests]

    pairs = [(m.sdd_size(), vl) for (m,_,_,vl,_) in bests]
    sizes, lls = zip(*pairs)

    best_tf = np.argmax([tf for (_,_,tf,_,_) in bests])
    best_vf = np.argmax([vf for (_,vf,_,_,_) in bests])
    best_tl = np.argmax([tl for (_,_,_,_,tl) in bests])
    best_vl = np.argmax([vl for (_,_,_,vl,_) in bests])

    plt.plot(sizes, lls, "kd")
    plt.plot(sizes[best_tf], lls[best_tf], "bo", label="best TF")
    plt.plot(sizes[best_vf], lls[best_vf], "go", label="best VF")
    plt.plot(sizes[best_tl], lls[best_tl], "mo", label="best TL")
    plt.plot(sizes[best_tf], lls[best_tf], "yo", label="best TF")

    plt.legend()

    plt.savefig(os.path.join(save_path, "size_ll_plt"))

    plt.clf()
    plt.cla()
    plt.close()

    plt.subplot(2,1,1)
    plt.plot(range(1, len(efficiencies) + 1), efficiencies, "ro", label="Efficiency of best (TF)")
    plt.xlabel("Generation")
    plt.ylabel("Efficieny")
    plt.legend(loc="lower right")

    plt.subplot(2,1,2)
    plt.plot(range(1, len(v_lls) + 1), v_lls, label="LL (V) of best (TF)")
    plt.xlabel("Generation")
    plt.ylabel("LL")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_path, "efficieny_best"))

    plt.clf()
    plt.cla()
    plt.close()

def BEST_MODEL(log, save_path):

    models = log.get_prop("best_model")

    best_model_fit_t = np.argmax([fit_t for (model, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_fit_v = np.argmax([fit_v for (model, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_ll_t = np.argmax([ll_t for (model, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_ll_v = np.argmax([ll_v for (model, fit_v, fit_t, ll_v, ll_t) in models])

    ind = [
        ("Fit (T)", best_model_fit_t),
        ("Fit (V)", best_model_fit_v),
        ("LL (T)", best_model_ll_t),
        ("LL (V)", best_model_ll_v)
    ]


    attrs = [(
    "According to", "Generation", "SDD Size", "Fitness (V)", "Fitness (T)", "LL (V)", "LL (T)"
    )]

    for i in ind:
        name, index  = i
        (model, fv, ft, lv, lt) = models[index]
        size = model.sdd_size()

        row = (name, index, size, fv, ft, lv, lt)

        attrs.append(row)

    print(stru.pretty_print_table(attrs))

    f = open(os.path.join(save_path, "best_models_stats"), "w")

    f.write(stru.pretty_print_table(attrs))
    f.close()

    f = open(os.path.join(save_path, "Best_models"), "w")

    for (name, index) in ind:
        f.write(f"Best model according to: {name} \n")
        f.write(models[index][0].to_string())
        f.write("\n\n\n\n")

    f.close()



def INDICATOR_PROFILE(log, save_path):
    profile = log.get_prop("indicator_profile")
    n_gen = len(profile)

    top_points = {}
    top_x = 5

    for gen in range(n_gen):
        #1) extract:
        #   -> indicators (always the same)
        #   -> amount of times used
        indicators, amounts = profile[gen]

        #2) find the top x indices for each gen
        top_indices_of_gen = np.argsort(amounts)[-top_x:]

        #3) collect the points for each indicator
        for top_index in top_indices_of_gen:
            indicator = indicators[top_index]
            amount = amounts[top_index]

            if indicator not in top_points:
                top_points[indicator] = []

            top_points[indicator] += [(gen, amount)]


    graphs = []
    # Now to connect the points with neighbouring generations
    for (key, value) in top_points.items():
        # key: indicator
        # value: [(gen, amount)]
        new_graphs = link_generations(value)

        graphs += new_graphs

    for graph in graphs:
        x, y = zip(*graph)
        plt.plot(x, y)


    plt.xlabel("Generation")
    plt.ylabel("Amount of indicator use" )

    plt.savefig(os.path.join(save_path, "indicator_profile"))

    plt.clf()
    plt.cla()
    plt.close()

    pass

def link_generations(points):
    # Points: [(gen, amount)]

    # graphs: [[(g1, a1), ... (gn, an)], ... [...]]
    #   -> list of lists
    #   -> each list is a list of (g, a) tuples where the gn = gm + 1
    graphs = []
    current_graph = [points[0]]

    for i in range(1, len(points)):
        pr_g, _ = current_graph[-1]
        c_g, c_a = points[i]

        if c_g == pr_g + 1:
            current_graph += [(c_g, c_a)]
        else:
            graphs += [current_graph]
            current_graph = [(c_g, c_a)]

    graphs += [current_graph]

    return graphs


def LIVE_DEAD_SIZE(log, save_path):
    live = log.get_prop("live_size")
    dead = log.get_prop("dead_size")

    plt.plot(range(len(live)), live, label= "Live size")
    plt.plot(range(len(dead)), dead, label= "Dead size")

    plt.xlabel("Generation")
    plt.ylabel("Size")

    plt.legend(loc="bottom right")
    plt.savefig(os.path.join(save_path, "live_dead"))

    plt.clf()
    plt.cla()
    plt.close()

def NB_FACTORS(log, save_path):
    dat = log.get_prop("nb_factors")

    s_min = [min(gen) for gen in dat]
    s_max = [max(gen) for gen in dat]
    s_avg = [np.mean(gen) for gen in dat]
    s_best = log.get_prop("best_nb_factors")

    plt.plot(range(len(s_min)), s_min, label = "Min nb fact.")
    plt.plot(range(len(s_min)), s_max, label = "Min nb fact.")
    plt.plot(range(len(s_min)), s_avg, label = "Min nb fact.")
    plt.plot(range(len(s_min)), s_best, label= "Best model")

    plt.xlabel("Generation")
    plt.ylabel("Number")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "nb_factors"))

    plt.clf()
    plt.cla()
    plt.close()

def BEST_IND(log, save_path):
    best_fits = log.get_prop("fit_t")
    best_inds = log.get_prop("best_ind")


    max_gen = np.argmax([max(gen) for gen in best_fits])

    print(len(best_inds))
    print(len(best_fits))
    print(max_gen)
    print(best_inds[max_gen])

def BEST_AND_SIZE(log, save_path):
    size = log.get_prop("size")
    fit_t = log.get_prop("fit_t")
    fit_v = log.get_prop("fit_v")
    ll_t = log.get_prop("ll_t")
    ll_v = log.get_prop("ll_v")

    top_ind_ll_t = [np.argmax(gen) for gen in ll_t]
    top_ind_fit_v = [np.argmax(gen) for gen in fit_v]

    top_ll_t = [max(gen) for gen in ll_t]
    top_ll_v = [gen[top_ind_ll_t[i]] for (i, gen) in enumerate(ll_v)]
    top_sizes = [gen[top_ind_ll_t[i]] for (i, gen) in enumerate(size)]

    top_fit_v = [max(gen) for gen in fit_v]
    top_ll_v_ = [gen[top_ind_fit_v[i]] for (i, gen) in enumerate(ll_v)]
    top_sizes = [gen[top_ind_ll_t[i]] for (i, gen) in enumerate(size)]

    plt.plot(range(len(fit_t)), top_ll_t, label="LL(T) of Best according to T")
    plt.plot(range(len(fit_t)), top_ll_v, label="LL(V) of Best according to T")

    # TODO: hacky code
    if "base_ll_tr" in log:
        goal_ll_t = log.get_prop("base_ll_tr")[0]
        plt.plot(range(len(fit_t)), [goal_ll_t for i in range(len(fit_t))], label="Goal ll (T)")

    plt.xlabel("Generation")
    plt.ylabel("LL")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "best_ll"))

    plt.clf()
    plt.cla()
    plt.close()


    plt.subplot(2,1, 1)
    plt.plot(range(len(fit_t)), top_fit_v, label="FIT(V) of Best according to V")

    #TODO: hacky code
    if "base_fit_va" in log:
        goal_fit_v = log.get_prop("base_fit_va")[0]
        plt.plot(range(len(fit_t)), [goal_fit_v for i in range(len(fit_t))], label="Goal fit (V)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")

    plt.subplot(2,1, 2)
    plt.plot(range(len(fit_t)), top_ll_v_, label="LL(V) of Best according to V")

    #TODO: hacky code
    if "base_ll_va" in log:
        goal_ll_v = log.get_prop("base_ll_va")[0]
        plt.plot(range(len(fit_t)), [goal_ll_v for i in range(len(fit_t))], label="Goal ll (V)")

    plt.xlabel("Generation")
    plt.ylabel("LL")
    plt.legend(loc = "lower right")

    plt.savefig(os.path.join(save_path, "best_fit"))

    plt.clf()
    plt.cla()
    plt.close()

def SIZES(log, save_path):
    sizes = log.get_prop("sizes")

    size_best = log.get_prop("best_size")

    max_s = [max(it) for it in sizes]
    min_s = [min(it) for it in sizes]
    avg_s = [np.mean(it) for it in sizes]

    plt.plot(range(len(sizes)), max_s, label="Max size per iteration")
    plt.plot(range(len(sizes)), min_s, label="Min size per iteration")
    plt.plot(range(len(sizes)), avg_s, label="Average size per iteration")
    plt.plot(range(len(size_best)), size_best, label="Best model size (according to fitness)")

    # TODO: hacky code
    if "base_size" in log:
        base_s = log.get_prop("base_size")[0]
        plt.plot(range(len(sizes)), [base_s for i in range(len(sizes))], label="Goal size of SDD")

    plt.xlabel("Generation")
    plt.ylabel("Size")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "Sizes"))

    plt.clf()
    plt.cla()
    plt.close()



def TIMES(log, save_path):
    t_fit = np.mean(log.get_prop("time: fitting"))
    t_mut = np.mean(log.get_prop("time: mutation"))

    # Create time plot
    times = log.get_prop("time")
    times_fitting = log.get_prop("time: fitting")
    times_cross = log.get_prop("time: crossing")
    times_mutation = log.get_prop("time: mutation")
    times_extra = log.get_prop("time: extra")

    plt.plot(range(len(times)), times, label="Times per iteration")
    plt.plot(range(len(times)), times_fitting, label="Times per iteration (fitting)")
    plt.plot(range(len(times)), times_cross, label="Times per iteration (cross)")
    plt.plot(range(len(times)), times_mutation, label="Times per iteration (mutation)")
    plt.plot(range(len(times)), times_extra, label="Extra time")
    plt.xlabel("Generation")
    plt.ylabel("Time (s)")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "Times"))

    plt.clf()
    plt.cla()
    plt.close()

    print(t_fit, t_mut)

def FITNESS_VALID(log, save_path):


    max_fit_v = [max(it) for it in log.get_prop("fit_v")]
    avg_fit_v = [np.mean(it) for it in log.get_prop("fit_v")]
    min_fit_v = [min(it) for it in log.get_prop("fit_v")]

    plt.plot(range(1, len(avg_fit_v) + 1), avg_fit_v, label = "populaion fitness (V)")
    plt.plot(range(1, len(max_fit_v) + 1), max_fit_v, label = "maximum fitness (V)")
    plt.plot(range(1, len(min_fit_v) + 1), min_fit_v, label = "minimum fitness (V)")

    if "base_fit_va" in log:
        goal_fit_v = log.get_prop("base_fit_va")[0]
        plt.plot(range(1, len(max_fit_v) + 1), [goal_fit_v for i in range(len(max_fit_v))], label="Goal (V)")


    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "validation_fit_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()

def FITNESS_TRAIN(log, save_path):

    max_fit_t = [max(it) for it in log.get_prop("fit_t")]
    avg_fit_t = [np.mean(it) for it in log.get_prop("fit_t")]
    min_fit_t = [min(it) for it in log.get_prop("fit_t")]

    plt.plot(range(1, len(avg_fit_t) + 1), avg_fit_t, label = "populaion fitness (T)")
    plt.plot(range(1, len(max_fit_t) + 1), max_fit_t, label = "maximum fitness (T)")
    plt.plot(range(1, len(min_fit_t) + 1), min_fit_t, label = "minimum fitness (T)")

    if "base_fit_tr" in log:
        goal_fit_t = log.get_prop("base_fit_tr")[0]
        plt.plot(range(1, len(max_fit_t) + 1), [goal_fit_t for i in range(len(max_fit_t))], label="Goal (T)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "training_fit_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()

def LL_TRAIN(log, save_path):


    min_ll_t = [min(it) for it in log.get_prop("ll_t")]
    max_ll_t = [max(it) for it in log.get_prop("ll_t")]
    avg_ll_t = [np.mean(it) for it in log.get_prop("ll_t")]

    plt.plot(range(1, len(min_ll_t) + 1), min_ll_t, label="min LL (T)")
    plt.plot(range(1, len(avg_ll_t) + 1), avg_ll_t, label="pop LL (T)")
    plt.plot(range(1, len(max_ll_t) + 1), max_ll_t, label="max LL (T)")

    if "base_ll_tr" in log:
        goal_ll_t = log.get_prop("base_ll_tr")[0]
        plt.plot(range(1, len(max_ll_t) + 1), [goal_ll_t for i in range(len(max_ll_t))], label="Goal (T)")

    plt.xlabel("Generation")
    plt.ylabel("Log Liklihood")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "training_ll_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()


def LL_VALID(log, save_path):

    min_ll_v = [min(it) for it in log.get_prop("ll_v")]
    max_ll_v = [max(it) for it in log.get_prop("ll_v")]
    avg_ll_v = [np.mean(it) for it in log.get_prop("ll_v")]

    plt.plot(range(1, len(min_ll_v) + 1), min_ll_v, label="min LL (V)")
    plt.plot(range(1, len(avg_ll_v) + 1), avg_ll_v, label="pop LL (V)")
    plt.plot(range(1, len(max_ll_v) + 1), max_ll_v, label="max LL (V)")

    if "base_ll_va" in log:
        goal_ll_v = log.get_prop("base_ll_va")[0]
        plt.plot(range(1, len(max_ll_v) + 1), [goal_ll_v for i in range(len(max_ll_v))], label="Goal (V)")

    plt.xlabel("Generation")
    plt.ylabel("Log Liklihood")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, 'validation_ll_plot.pdf'))

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
    gen = recursive_generator(n_vars, imgr, mgr, max_nb_factors = local_max)

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
    gen = recursive_generator(n_vars, imgr, fn_sparseness=0.005)

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
    gen = recursive_generator(vrs, imgr,operation_distr = [10, 10, 1])
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

def script5():
    a = disj([lit(1), lit(2)])
    b = neg(a)

    all_worlds = [
    [1,1],
    [1,0],
    [0,1],
    [0,0]
    ]

    print(b.to_string())
    print([a.evaluate(w) for w in all_worlds])
    print([b.evaluate(w) for w in all_worlds])

    mgr = SddManager(14)
    imgr = indicator_manager([range(4,5), range(12, 15)])

    c = conj([lit(1), lit(2)])
    d = lit(3)
    e = impl(c, d)
    n = neg(e)

    print(e.to_string())
    print(n.to_string())
    print(e.evaluate([1,1,0]))
    print(n.evaluate([1,1,0]))

    modl = Model(3, imgr, mgr)
    modl.update_sdd()

    modl.add_factor(e, 2)

    modl.update_sdd()
    modl.partition()

    print(modl.to_string())

def test_script():
    n_vars = 2
    global_max = 400
    local_max = 200
    bot = n_vars + 1
    top = bot + global_max

    mgr = SddManager(top)
    imgr  = indicator_manager(range(bot, top))
    cmgr = count_manager()

    modl = Model(n_vars, imgr, mgr, cmgr)
    modl.add_factor(conj([lit(1), lit(2)]), 2)

    modl.update_sdd()
    modl.partition()

    print(modl.to_string())
    print(imgr.get_indicator_counts())

    clone = modl.clone()

    print(clone.to_string())
    print(imgr.get_indicator_counts())

    modl.remove_factor(modl.get_features()[0])
    modl.partition()

    print(imgr.get_indicator_counts())
    print(modl.to_string())
    print(clone.to_string())



if __name__ == "__main__":
    __main__()
