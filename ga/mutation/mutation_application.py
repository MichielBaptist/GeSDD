from pysdd.sdd import Vtree, SddManager
from model.model import Model
import random
import numpy as np
import time
from functools import reduce
import utils.time_utils as tut

from logic.conj import conj
from logic.disj import disj
from logic.equiv import equiv
from logic.cons import cons
from logic.lit import lit
from logic.neg import neg

import math
import multiprocessing as mp
import os
import utils.string_utils as stru

n_cores = 16

def mutate_population(population, mutation, mpr, train):
    # 1) Make sure only roughly mpr% of models get mutated
    m_ind, m_pop = filter_pop(population, mpr)

    t = tut.get_timer()

    # 2) Apply the mutation across filtered models
    m_pop = [apply_mutation_series(mdl, mutation, t, train) for mdl in m_pop]

    t.summary("Summary of mutation")

    # 3) Place the mutated models back in the list
    for i, mdl in zip(m_ind, m_pop):
        population[i] = mdl

    return population


# TODO: implement multiprog mutation (if needed)
def mutate_population_multiprog(population, mutation, mpr, run_name, train):
    t = tut.get_timer()
    t.start()
    # 1) Make sure only roughly mpr% of models get mutated
    m_ind, m_pop = filter_pop(population, mpr)
    t.t("Filter pop")

    # 2) Apply the mutations to the models only
    #    this step does not involve parallel programming
    #    and is used to alter common datastructures.
    #    This obtains a list of arguments needed to apply
    #    the actual mutation to the SDD.
    sdd_args = [mutation.apply(mdl, train) for mdl in m_pop]

    t.t("Applying model")

    models, sdd_args = zip(*sdd_args)

    tmp_dir = "ga/mutation/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    run_tmp = os.path.join(tmp_dir, run_name)
    if not os.path.exists(run_tmp):
        os.makedirs(run_tmp)

    # 3) Save the SDD and vtrees to files
    paths = [save_sdd_and_vtree(mdl, run_tmp, i) for i, mdl in enumerate(m_pop)]

    t.t("Saving sdds/vtrs")

    # 4) Multiprog mutate population
    m_pop = mutate_pop_multi(m_pop, paths, sdd_args, mutation)

    t.t("Mutation multiprog")

    # 5) Place the mutated models back in the list
    for i, mdl in zip(m_ind, m_pop):
        population[i] = mdl

    t.summary("Summary of mutation")
    return population

def mutate_pop_multi(pop, paths, args, mutation):

    sdd_paths, vtr_paths = zip(*paths)
    args_names = [
        "sdd_path",
        "vtr_path",
        "mutation",
        "args",
        "t1"
    ]
    model_args = zip(
        sdd_paths,
        vtr_paths,
        [mutation for i in range(len(sdd_paths))],
        args,
        [time.time() for i in range(len(sdd_paths))]
    )
    model_args = [dict(zip(args_names, model)) for model in model_args]

    pp.map(mutate_model, model_args)

    # 5) reload the sdds/vtrs
    new_mgrs = []
    new_sdds = []
    for (sdd_path, vtr_path) in paths:
        new_mgr = load_manager(vtr_path)
        new_sdd = load_sdd(sdd_path, new_mgr)
        new_mgrs.append(new_mgr)
        new_sdds.append(new_sdd)

    for (mdl, new_sdd, new_mgr) in zip(pop, new_sdds, new_mgrs):
        mdl.sdd = new_sdd
        mdl.mgr = new_mgr

    return pop


def mutate_model(args):
    print(f"Loading took: {time.time() - args['t1']}")
    sdd_path = args["sdd_path"]
    vtr_path = args["vtr_path"]
    mutation = args["mutation"]
    mut_args = args["args"]

    mgr = load_manager(vtr_path)
    old_sdd = load_sdd(sdd_path, mgr)

    new_sdd = mutation.applySDD_wrap(old_sdd, mgr, mut_args)

    save_vtr_path(mgr.vtree(), vtr_path)
    save_sdd_path(new_sdd, sdd_path)

    #pp.map(mutate_mdl_wrap(p, a)

#--------------------------------------- SAVE ----------------------------------

def load_sdd(sdd_path, mgr):
    lsdd = mgr.read_sdd_file(sdd_path.encode())
    lsdd.ref()
    return lsdd

def load_manager(vtr_path):
    vtr = Vtree.from_file(vtr_path)
    mgr = SddManager.from_vtree(vtr)
    mgr.auto_gc_and_minimize_on()
    return mgr

def save_sdd_and_vtree(mdl, dir, num):
    sdd = mdl.sdd
    vtr = mdl.mgr.vtree()
    sp =save_sdd(sdd, dir, num)
    vp =save_vtr(vtr, dir, num)
    return (sp,vp)

def save_sdd(sdd, dir, num):
    file = os.path.join(dir, f"sdd_{num}")
    return save_sdd_path(sdd, file)

def save_sdd_path(sdd_, path):
    sdd_.save(path.encode())
    return path

def save_vtr(vtr, dir, num):
    file = os.path.join(dir, f"vtr_{num}")
    return save_vtr_path(vtr, file)

def save_vtr_path(vtr, path):
    vtr.save(path.encode())
    return path


# -------------------------------------- MISC ----------------------------------

def apply_mutation_series(model, mutation, t, train):
    t.start()
    # 1) Apply to the model first
    model, args = mutation.apply(model, train)
    mgr = model.mgr

    t.t("Apply model")

    # 2) Apply to the actual SDD
    #    It is expected of the applySDD method that it always returns an
    #    sdd which is referenced. The method also expects an SDD which is referenced
    #    i.e. model.sdd should always be ref()d.
    sdd = mutation.applySDD_wrap(model.sdd, mgr, args)

    t.t("Apply SDD")

    # 3) Place new SDD back in the model
    model.set_compiled_sdd(sdd)

    return model

def should_mutate(mpr):
    return np.random.rand() < mpr

def filter_pop(population, mpr):
    filtered_pop = [(ind, mdl) for ind, mdl in enumerate(population) if should_mutate(mpr)]
    return zip(*filtered_pop)

pp = mp.Pool(n_cores)
