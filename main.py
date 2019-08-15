from model.model import Model
from logic import *
from gen.gen import random_generator, data_seeded_generator

from mpl_toolkits.mplot3d import Axes3D

import multiprocessing as mp
import math
import random
import gen.gen as g
import pysdd
import numpy as np
import sys
import os
import utils.time_utils as tut
from model_IO import model_io
import argparse

from Saver import saver

import data_gen

from graphviz import Source


from pysdd.sdd import SddManager, Vtree
from model.indicator_manager import indicator_manager
from model.count_manager import count_manager
from datio import IO_manager

import ga.selection as ga_selection
import ga.crossover.crossover as ga_crossover
import ga.mutation.mutation as ga_mutation
import ga.fitness as ga_fitness

#import algorithm as ga_algorithm
import algorithm_multi as ga_algorithm
from algorithm import logbook

from logic.conj import conj
from logic.disj import disj
from logic.lit  import lit
from logic.cons import cons
from logic.factor import factor
from logic.equiv import equiv
from logic.neg import neg
from logic.impl import impl

from matplotlib import pyplot as plt

import time
import utils.string_utils as stru

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='GeSDD')
parser.add_argument('-train', required = True, help='The train data set')
parser.add_argument('-valid', required = True, help='The validation data set')
parser.add_argument('-run_folder',
                    required = False,
                    help='Name of the folder where tmp results are stored.',
                    default = "current_runs")
parser.add_argument('-run_name',
                    required = False,
                    help='Name of the run, used for creating intermediate folders.',
                    default = "default_run")
parser.add_argument('-alpha',
                    help='Decides how much SDD compactness to be preferred over fit.',
                    action = 'store',
                    type = float,
                    default = 5e-5
                    )
parser.add_argument('-population_size',
                    help='The size of the population.',
                    action='store',
                    type = int,
                    default=52)
parser.add_argument('-n_gens',
                    help='Determines the number of generations GeSDD will be run for.',
                    action = 'store',
                    type = int,
                    default = 30)
parser.add_argument('-gamma_mutations',
                    help='Determines the size of the mutation jumps.',
                    action = 'store',
                    type = float,
                    default = 0.05)
parser.add_argument('-candidate_size',
                    help='Determines the size of the mutation jumps.',
                    action = 'store',
                    type = int,
                    default = 4)
parser.add_argument('-gamma_cross',
                    help='Determines the size of the crossover jumps.',
                    action = 'store',
                    type = float,
                    default = 0.05)
parser.add_argument('-threshold',
                    help = 'Determines the threshold for removing features.',
                    action = 'store',
                    type = float,
                    default=0.1)
parser.add_argument('-seed',
                    help='Seed to be used by GeSDD',
                    action='store',
                    type = int,
                    default = None)
parser.add_argument('-max_nb_f',
                    help='The maximum number of features a MLN can have.',
                    action='store',
                    type = int,
                    default=100)
parser.add_argument('-generator',
                    help='The type of rule generation to be used.',
                    action='store',
                    required = True,
                    choices=['random', 'seeded'],
                    default='seeded')
parser.add_argument('-tournament_size',
                    help='The size of a tournament round for the tournament selection.',
                    action='store',
                    type = int,
                    default=5)
parser.add_argument('-cnt',
                    choices = ["yes", "no"],
                    help='Should a run be contunued? When true it will continue the run in run_dir/run_name/',
                    action='store',
                    default="no")

def __main__(args):

    # Generate a random seed
    seed = args.seed
    if seed == None:
        seed = random.randrange(2**20)
    print("Seed was:", seed)

    np.random.seed(seed)
    random.seed(seed)

    print("Using data files!")
    train_path = args.train
    valid_path = args.valid
    train, n_worlds_train, n_vars = IO_manager.read_from_csv(train_path, ',')
    valid, n_worlds_valid, n_vars = IO_manager.read_from_csv(valid_path, ',')
    n_worlds = n_worlds_train + n_worlds_valid
    test_train_ratio = n_worlds_valid / n_worlds

    global_max = args.max_nb_f
    bot = n_vars + 1
    top = bot + global_max
    imgr  = indicator_manager(range(bot, top))
    cmgr = count_manager()

    train = cmgr.compress_data_set(train, "train")
    valid = cmgr.compress_data_set(valid, "valid")

    # doesn't use imgr
    if args.generator == 'seeded':
        gen = data_seeded_generator(train,  None, cmgr, top)
    elif args.generator == 'random':
        gen = random_generator(n_vars, top, cmgr)

    empty_model = Model(n_vars,
                        manager = SddManager(top),
                        count_manager=cmgr)
    empty_model.compile()
    empty_model.fit(train, "train")
    empty_ll = empty_model.LL(train, "train")
    print(f"Empty model LL: {empty_ll}")

    params = {}
    params['run_folder'] = args.run_folder
    params['run_name'] = args.run_name
    params['model_io'] = model_io(args.run_folder, args.run_name)
    params['empty_model'] = empty_model
    params['tmp_path'] = os.path.join("tmp_out", args.run_name)
    params['seed'] = seed
    params['selection'] = ga_selection.tournament_selection(args.tournament_size)
    params['paring'] = ga_selection.pairing()
    params['cross'] = ga_crossover.rule_trade_cross(args.run_name, nb_rules_pct = args.gamma_cross)
    """
    params['mutation'] = gao.script_mutation([
                                             gao.threshold_mutation(0.1),
                                             gao.multi_mutation([
                                                    gao.add_mutation(),
                                                    gao.weighted_remove_mutation(),
                                                    gao.rule_expansion_mutation(),
                                                    gao.rule_shrinking_mutation(),
                                                    gao.sign_flipping_mutation()
                                                ]),
                                             ])"""
    params['mutation'] = ga_mutation.script_mutation([
        #ga_mutation.add_mutation(gen),
        ga_mutation.threshold_mutation(args.threshold),
        ga_mutation.multi_mutation([
            ga_mutation.apmi_add_pct_mutation(gen, k = args.candidate_size, pct = 0.05),
            ga_mutation.remove_pct_mutation(pct = args.gamma_mutations),
            ga_mutation.apmi_feat_flip_pct_mutation_global(pct = args.gamma_mutations, ratio = 0.5, k = args.candidate_size),
            ga_mutation.apmi_feat_flip_pct_mutation(pct = args.gamma_mutations, ratio = 0.5, k = args.candidate_size),
            ga_mutation.apmi_feat_shrink_pct_mutation(pct = args.gamma_mutations, k = args.candidate_size),
            ga_mutation.apmi_feat_expand_pct_mutation(pct = args.gamma_mutations, k = args.candidate_size)
        ])
    ])

    #params['mutation'] = ga_mutation.add_mutation(gen)
    params['fitness'] = ga_fitness.fitness5(args.alpha, empty_ll)
    #params['fitness'] = ga_fitness.globalFit(0.005)
    params['generator'] = gen
    params['logbook'] = logbook()
    params['pop_size'] = args.population_size
    params['train'] = train
    params['valid'] = valid
    params['mutate_probability'] = 0.8
    params['n_gens'] = args.n_gens
    params['it_before_restart'] = args.n_gens
    params['n_select'] = int(args.population_size/2)
    params['indicator_manager'] = imgr
    params['n_worlds'] = n_worlds
    params['n_vars'] = n_vars
    params['test_train_ratio'] = test_train_ratio
    params['empty_ll'] = empty_ll
    params['current_best'] = -1
    params['train_file'] = args.train
    params['valid_file'] = args.valid
    params['candidate_size'] = args.candidate_size

    pop, params= ga_algorithm.run(params, cnt = (args.cnt == "yes"))

    zero_t_ll = empty_model.LL(train, "train")
    zero_v_ll = empty_model.LL(valid, "valid")
    zero_t_fit = params['fitness'].of(empty_model, train, "train")
    zero_v_fit = params['fitness'].of(empty_model, valid, "valid")

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
        #LIVE_DEAD_SIZE,
        INDICATOR_PROFILE,
        BEST_MODEL,
        MODEL_EFFICIENY,
        FEATURE_SIZES
    ]

    svr = saver("run")
    svr.save_run(params, logbook, aggregators)

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
    f7 = conj([disj([a, _b]), impl(_f, g)])

    model.add_factor(f1, 3)
    model.add_factor(f2, 1)
    model.add_factor(f3, -2)
    model.add_factor(f4, 2)
    model.add_factor(f5, 1.5)
    model.add_factor(f6, -1)
    model.add_factor(f7, 2)

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
    f7 = impl(disj([neg(lit(2), conj([lit(-1), lit(3)]))]), conj([disj([lit(5), lit(-2)]),lit(4)]))

    #f6 = disj([])

    model.add_factor(f1, 2)
    model.add_factor(f2, 2)
    model.add_factor(f3, 3)
    model.add_factor(f4, 3)
    model.add_factor(f5, 1)
    model.add_factor(f6, 2)
    #model.add_factor(f7, 2.5)

    return model

def custom_model6(model):
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
    f7 = impl(disj([neg(lit(2)), conj([lit(-1), lit(3)])]), conj([disj([lit(5), lit(-2)]),lit(4)]))

    #f6 = disj([])

    model.add_factor(f1, 2)
    model.add_factor(f2, 2)
    model.add_factor(f3, 3)
    model.add_factor(f4, 3)
    model.add_factor(f5, 1)
    model.add_factor(f6, 2)
    model.add_factor(f7, 2.5)

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


def FEATURE_SIZES(log, save_path):
    best_feature_sz = log.get_prop("best_feature_size")
    feature_sz = log.get_prop("feature_sizes")

    avg_size_pop = [np.mean([np.mean(mdl) for mdl in gen]) for gen in feature_sz]
    avg_size_best = [np.mean(gen) for gen in best_feature_sz]

    plt.plot(range(len(avg_size_pop)), avg_size_pop, label ="Average feature size pop")
    plt.plot(range(len(best_feature_sz)), avg_size_best, label="average feature size best")

    plt.xlabel("generation")
    plt.ylabel("Average feature length.")
    plt.legend()
    plt.savefig(os.path.join(save_path, "average_feat_sz"))

    plt.clf()
    plt.cla()
    plt.close()

def MODEL_EFFICIENY(log, save_path):
    bests = log.get_prop("best_model")
    zero_v_ll = log.get_point(0, "zero_v_ll")

    efficiencies = [ s / (vl - zero_v_ll) for (s, _, _, vl,_) in bests]
    v_lls = [vl for (_,_,_,vl,_) in bests]

    pairs = [(s, vl) for (s,_,_,vl,_) in bests]
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

    best_model_fit_t = np.argmax([fit_t for (size, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_fit_v = np.argmax([fit_v for (size, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_ll_t = np.argmax([ll_t for (size, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_ll_v = np.argmax([ll_v for (size, fit_v, fit_t, ll_v, ll_t) in models])

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
        (size, fv, ft, lv, lt) = models[index]

        row = (name, index, size, fv, ft, lv, lt)

        attrs.append(row)

    print(stru.pretty_print_table(attrs))

    f = open(os.path.join(save_path, "best_models_stats"), "w")

    f.write(stru.pretty_print_table(attrs))
    f.close()

    f = open(os.path.join(save_path, "Best_models"), "w")

    for (name, index) in ind:
        f.write(f"Best model according to: {name} \n")
        #f.write(models[index][0].to_string())
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

def ref_test():
    mgr = SddManager(5)
    eq = equiv([conj([lit(1), disj([lit(2), lit(3)])]), conj([lit(2), lit(3), lit(4), lit(5)])])
    eq = conj([
        neg(conj([neg(lit(1)), lit(2)])),
        conj([lit(3), lit(4)])
    ])
    sdd = eq.compile(mgr)

    print(sdd.garbage_collected())

    print(sdd.size())
    print(mgr.live_size())
    print(mgr.dead_size())

def mi_test():

    def mrmr(lst_candidate, lst_present, data, cmgr):
        s = len(lst_present)
        feature_str = loo_MI(lst_candidate, data, cmgr)

        print(f"Strength: {lst_candidate} --> {feature_str}")
        print("Present:")
        for p in lst_present:
            print(p)
            feature_red = feature_redundancy(conj(lst_candidate), lst_present, data, cmgr)
        print(f"Redundancy: {feature_red}")
        print(f"mrmr: {feature_str - feature_red}")
        return feature_red
        return feature_str - feature_red/s

    def feature_redundancy(f, lst, data, cmgr):
        return sum([pmi(f, present, data, cmgr) for present in lst])/len(lst)

    def pairwise_single_out(lst):
        return [(lst[i], lst[:i]+lst[i+1:]) for i in range(len(lst))]

    def loo_MI(lst, data, cmgr):
        pairs = pairwise_single_out(lst)
        pmis = [pmi(p[0], conj(p[1]), data, cmgr) for p in pairs]
        return sum(pmis)/len(pmis)

    def all_pairs(lst):
        if len(lst) == 1:
            return []

        head = lst[0]
        tail = lst[1:]

        pairs = [(head, r) for r in tail]
        rest_pairs = all_pairs(tail)

        return pairs + rest_pairs

    def cov(l, r, data, cmgr):
        return cmgr.count_factor(conj([l, r]), data) - cmgr.count_factor(l, data)*cmgr.count_factor(l, data)

    def apcov(lst, data, cmgr):
        pairs = all_pairs(lst)
        pcovs = [cov(p[0], p[1], data, cmgr) for p in pairs]
        return sum(pcovs)/len(pcovs)

    def apmi(lst, data, cmgr):
        pairs = all_pairs(lst)
        pmis = [pmi(p[0], p[1], data, cmgr) for p in pairs]
        return sum(pmis)/len(pmis)

    def chain_MI(lst, dat, cmgr):
        if len(lst) == 2:
            return pmi(lst[0], lst[1], dat, cmgr)

        head = lst[0]
        tail = lst[1:]

        return pmi(head, conj(tail), dat, cmgr) + chain_MI(tail, dat, cmgr)

    def pmi(l, r, data, cmgr):
        def ent(ps):
            if 0 in ps:
                return 0
            return -sum([p*math.log2(p) for p in ps])

        def entropy(f, dat, cmgr):
            p = cmgr.count_factor(f, dat)
            np= cmgr.count_factor(neg(f), dat)
            return ent([p, np])

        def joint_entropy(l, r, dat, cmgr):
            ps = [conj([l,r]), conj([l, neg(r)]), conj([neg(l), r]), conj([neg(l), neg(r)])]
            ps = [cmgr.count_factor(p, dat) for p in ps]
            return ent(ps)

        Hl = entropy(l, data, cmgr)
        Hr = entropy(r, data, cmgr)
        Hlr= joint_entropy(l, r, data, cmgr)

        return Hl + Hr - Hlr

    data_files = ("data/nltcs/nltcs.train.wdata", "data/nltcs/nltcs.valid.wdata")
    #data_files = ("data/synthetic/synth_train_5_3.dat", "data/synthetic/synth_valid_5_3.dat")
    #data_files = ("data/synthetic/synth_train_8_4.dat", "data/synthetic/synth_valid_8_4.dat")

    train_path, valid_path = data_files
    train, n_worlds_train, n_vars = IO_manager.read_from_csv_num(train_path, ',')
    valid, n_worlds_valid, n_vars = IO_manager.read_from_csv_num(valid_path, ',')
    n_worlds = n_worlds_train + n_worlds_valid
    test_train_ratio = n_worlds_valid / n_worlds

    global_max = 70
    bot = n_vars + 1
    top = bot + global_max
    cmgr = count_manager()

    train = cmgr.compress_data_set(train)
    valid = cmgr.compress_data_set(valid)

    gen = random_generator(n_vars, top, cmgr, nb_threshold = 5)

    apmis = []
    apmis2= []
    mrmrs = []
    leave_one_out = []
    lls = []
    lls2 = []
    fs = []
    ws1=[]
    ws2=[]
    sz1=[]
    sz2=[]
    tmr = tut.get_timer()
    tmr.start()
    for i in range(500):
        print(f"Iteration: {i}")
        f = gen.random_feature_f()
        fs.append(f)
        lst = f.list_of_factors
        tmr.t("gen feat")
        #apmis.append(apmi(lst, train, cmgr))
        apmis.append(chain_MI(lst, train, cmgr))
        apmis2.append(apmi(lst, train, cmgr))
        leave_one_out.append(loo_MI(lst, train, cmgr))
        #apmis.append(apmi(lst, train, cmgr))
        tmr.t("Mut. info.")

        #mdl = Model(n_vars, manager = SddManager(n_vars + 1), count_manager = cmgr)
        mdl = gen.gen()
        present_feats = [f for (f,_,_,_) in mdl.get_features()]
        mrmrs.append(mrmr(lst, present_feats, train, cmgr))
        mdl.compile()
        #mdl = gen.gen()
        mdl.fit(train)
        ll1 = mdl.LL(train)
        mdl.add_factor(f,0)
        mdl.compile()
        sz1.append(mdl.sdd_size())
        mdl.fit(train, max_fun=20)
        ws1.append(mdl.get_factor_weights()[-1])
        lls.append(mdl.LL(train) - ll1)
        tmr.t("Fitting mdl")
        #print(mdl.to_string())
        #mdl.remove_factor(mdl.get_features()[-1])
        #print(mdl.to_string())
        #mdl2 = Model(n_vars, manager = SddManager(n_vars + 1), count_manager = cmgr)
        #mdl.add_factor(f, 0)
        #print(mdl.to_string())
        #mdl.compile()
        #sz2.append(mdl.sdd_size())
        #mdl.fit(train, max_fun=20)
        #lls2.append(mdl.LL(train) - ll1)
        #ws2.append(mdl.get_factor_weights()[-1])


    plt.plot([len(f.list_of_factors) for f in fs], lls, 'ro', label='Factor length vs lls')
    plt.show()
    plt.plot([len(f.list_of_factors) for f in fs], apmis, 'ro', label='Factor length vs CHain')
    plt.show()
    plt.plot([len(f.list_of_factors) for f in fs], apmis2, 'ro', label = 'Factor length vs APMI')
    plt.show()
    plt.plot([len(f.list_of_factors) for f in fs], leave_one_out, 'ro', label='Factor length vs LOO')
    plt.show()
    plt.plot([len(f.list_of_factors) for f in fs], mrmrs, 'ro', label='Factor length vs MRMR')
    plt.show()
    plt.plot(apmis, lls, 'ro')
    plt.xlabel("Chain")
    plt.ylabel("LLs")
    plt.show()
    plt.plot(apmis2, lls, 'ro')
    plt.xlabel("APMI")
    plt.ylabel("LLs")
    plt.show()
    plt.plot(leave_one_out, lls, 'ro')
    plt.xlabel("LOO")
    plt.ylabel("LLs")
    plt.show()
    plt.plot(mrmrs, lls, 'ro')
    plt.xlabel("MRMR")
    plt.ylabel("LLs")
    plt.show()
    #plt.plot(lls, lls2, "ro")
    #plt.xlabel("LLS1")
    #plt.ylabel("LLS2")
    #plt.show()
    #plt.plot(ws1, ws2, "ro")
    #plt.xlabel("WS1")
    #plt.ylabel("WS2")
    #plt.show()
    #plt.plot(ws1, lls, "ro")
    #plt.xlabel("WS1")
    #plt.ylabel("LLS1")
    #plt.show()
    #plt.plot(ws2, lls2, "ro")
    #plt.xlabel("WS2")
    #plt.ylabel("LLS2")
    #plt.show()
    #plt.plot(sz1, lls, 'ro')
    #plt.plot(sz2, lls2, 'bo')
    #plt.xlabel("SZ")
    #plt.ylabel("LLs")
    #plt.show()

    plt.plot(apmis, apmis2, 'ro',label = "Chain IG vs APMI")
    plt.show()
    plt.plot(apmis, leave_one_out, 'ro', label="Chain IG vs LOO")
    plt.show()
    plt.plot(apmis2, leave_one_out, 'ro', label="APMI vs LOO")
    plt.show()
    plt.plot(apmis2, apmis, 'ro', label="APMI vs CHAIN")
    plt.show()

    tmr.summary()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([len(f.list_of_factors) for f in fs], apmis,lls, marker = 'o')

    ax.set_xlabel('Sizes')
    ax.set_ylabel('Chain')
    ax.set_zlabel('lls')

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([len(f.list_of_factors) for f in fs], apmis2,lls, marker = 'o')

    ax.set_xlabel('Sizes')
    ax.set_ylabel('Apmis')
    ax.set_zlabel('lls')

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([len(f.list_of_factors) for f in fs], leave_one_out,lls, marker = 'o')

    ax.set_xlabel('Sizes')
    ax.set_ylabel('LOO')
    ax.set_zlabel('lls')

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([len(f.list_of_factors) for f in fs], mrmrs,lls, marker = 'o')

    ax.set_xlabel('Sizes')
    ax.set_ylabel('MRMR')
    ax.set_zlabel('lls')

    plt.show()
def img_vtree():
    vtree = Vtree(4)
    s = Source(vtree.dot())
    s.format = "png"
    s.render("vtree", view=True)

    vtree = Vtree(4, vtree_type="right")
    s = Source(vtree.dot())
    s.format = "png"
    s.render("vtree2", view=True)

def size_test():
    data_files = ("data/nltcs/nltcs.train.wdata", "data/nltcs/nltcs.valid.wdata")
    #data_files = ("data/synthetic/synth_train_5_3.dat", "data/synthetic/synth_valid_5_3.dat")
    #data_files = ("data/synthetic/synth_train_8_4.dat", "data/synthetic/synth_valid_8_4.dat")

    train_path, valid_path = data_files
    train, n_worlds_train, n_vars = IO_manager.read_from_csv_num(train_path, ',')
    valid, n_worlds_valid, n_vars = IO_manager.read_from_csv_num(valid_path, ',')
    n_worlds = n_worlds_train + n_worlds_valid
    test_train_ratio = n_worlds_valid / n_worlds

    global_max = 70
    bot = n_vars + 1
    top = bot + global_max
    cmgr = count_manager()

    train = cmgr.compress_data_set(train)
    valid = cmgr.compress_data_set(valid)

    gen = random_generator(n_vars, top, cmgr, nb_threshold = 10, max_feat_len = 8/16)
    gen2= random_generator(n_vars, top, cmgr, nb_threshold = 10, max_feat_len = 2/16)

    apmis = []
    lls = []
    lns = []
    tmr = tut.get_timer()
    tmr.start()
    empty = Model(n_vars, manager = SddManager(n_vars + 1), count_manager = cmgr)
    empty.compile()
    empty.fit(train)
    empty_ll = empty.LL(train)

    for i in range(100):
        print(i)
        mdl = gen.gen()
        print(mdl.nb_factors)
        mdl.fit(train)
        lls.append(mdl.LL(train) - empty_ll)
        lns.append(mdl.sdd_size())
    lls2 = []
    lns2 = []
    for i in range(100):
        print(i)
        mdl = gen2.gen()
        print(mdl.nb_factors)
        mdl.fit(train)
        lls2.append(mdl.LL(train) - empty_ll)
        lns2.append(mdl.sdd_size())

    plt.plot(lns, lls, "ro")
    plt.plot(lns2, lls2, "bo")
    plt.show()

def test_ll():
    test_dat = "data/nltcs/nltcs.test.wdata"
    train = "data/nltcs/nltcs.train.wdata"

    test, n_worlds_train, n_vars = IO_manager.read_from_csv_num(test_dat, ',')
    train, _, n_vars = IO_manager.read_from_csv_num(train, ',')
    cmgr = count_manager()
    test = cmgr.compress_data_set(test)
    train = cmgr.compress_data_set(train)

    models = mio.load_models()

    llsTrain = [mdl.LL(train) for mdl in models]
    llsTest = [mdl.LL(test) for mdl in models]

    print(stru.pretty_print_table(list(zip(llsTrain, llsTest))))

def mdl_test():
    mdl = Model(3, manager = SddManager(3 + 1))
    mdl.add_factor(conj([lit(1), lit(2), lit(3)]), 1)
    mdl.compile()
    mdl.partition()
    print(mdl.to_string())

def load_sets(args):

    print("Using data files!")
    train_path = args.train
    valid_path = args.valid
    #test_path = "data/nltcs/nltcs.test.data"
    train, n_worlds_train, n_vars = IO_manager.read_from_csv_num(train_path, ',')
    valid, n_worlds_valid, n_vars = IO_manager.read_from_csv_num(valid_path, ',')
    #test, n_worlds_test, n_vars = IO_manager.read_from_csv(test_path, ',')
    n_worlds = n_worlds_train + n_worlds_valid
    test_train_ratio = n_worlds_valid / n_worlds

    test, valid = train_test_split(valid, train_size = 0.5)

    data_gen.save("data/synthetic/test5", test, ",")
    data_gen.save(args.valid, valid, ",")
    data_gen.save(args.train, train, ",")

if __name__ == "__main__":
    args = parser.parse_args()
    #load_sets(args)
    __main__(args)
    #mdl_test()
    #mi_test()
    #img_vtree()
    #ref_test()
    #size_test()
    #test_ll()
