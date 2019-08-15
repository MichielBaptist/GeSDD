import os
import time
import multiprocessing as mp
from pysdd.sdd import Vtree, SddManager
n_cores = 10

def cross_pairs(pairs, cross):
    # pairs: [(left, right), ...] List of model pairs.
    # cross: A crossover operation.

    # 1) Cross only models. Each cross results in a (child1, child2, dict).
    # The resulting children are expected to be copies of their parents. They
    # should not use the same manager (i.e. the manager should also be copied)
    # This operation should take almost no time. (Can easily be done serial)
    children = [cross.apply(pair[0], pair[1]) for pair in pairs]

    # 2) aply the cross-over to the sdd's. Collect children
    # This operation may take considerable amount of time and could be
    # done in parallell for speedup.
    children_flat = []
    for (left, right, dict) in children:
        lSDD = cross.applyOneSDD(left.sdd, left.mgr, dict["largs"])
        rSDD = cross.applyOneSDD(right.sdd,right.mgr, dict["rargs"])

        # Make sure to not deref the sdd when
        # it doesn't change.
        if lSDD != left.sdd:
            left.sdd.deref()
        if rSDD != right.sdd:
            right.sdd.deref()

        left.set_compiled_sdd(lSDD)
        right.set_compiled_sdd(rSDD)

        children_flat.append(left)
        children_flat.append(right)

    return children_flat

def cross_pairs_multiprog(pairs, cross, run_name):
    # pairs: [(left, right), ...] List of model pairs.
    # cross: A crossover operation.

    # 1) Cross only models. Each cross results in a (child1, child2, dict).
    # The resulting children are expected to be copies of their parents. They
    # should not use the same manager (i.e. the manager should also be copied)
    # This operation should take almost no time. (Can easily be done serial)
    children = [cross.apply(pair[0], pair[1]) for pair in pairs]

    flat = []
    for (l, r, dict) in children:
        flat.append((l, dict["largs"]))
        flat.append((r, dict["rargs"]))

    models = [mdl for (mdl, args) in flat]
    args = [args for (mdl, args) in flat]

    paths = save_models([mdl for mdl in models], run_name)
    sdds = [s for (s,v) in paths]
    vtrs = [v for (s,v) in paths]

    children_applied = cross_multiprog(models, args, sdds, vtrs, cross)

    return children_applied

def cross_multiprog(pop, args, sdds, vtrs, cross):
    args_names = [
        "sdd_path",
        "vtr_path",
        "cross",
        "args",
        "t1"
    ]
    model_args = zip(
        sdds,
        vtrs,
        [cross for i in range(len(sdds))],
        args,
        [time.time() for i in range(len(sdds))]
    )
    model_args = [dict(zip(args_names, model)) for model in model_args]

    pp.map(cross_pair, model_args)

    # 5) reload the sdds/vtrs
    new_mgrs = []
    new_sdds = []
    for (sdd_path, vtr_path) in zip(sdds, vtrs):
        new_mgr = load_manager(vtr_path)
        new_sdd = load_sdd(sdd_path, new_mgr)
        new_mgrs.append(new_mgr)
        new_sdds.append(new_sdd)

    for (mdl, new_sdd, new_mgr) in zip(pop, new_sdds, new_mgrs):
        mdl.sdd = new_sdd
        mdl.mgr = new_mgr

    return pop

def cross_pair(args):
    sdd_path = args["sdd_path"]
    vtr_path = args["vtr_path"]
    cross = args["cross"]
    cross_args = args["args"]

    mgr = load_manager(vtr_path)
    old_sdd = load_sdd(sdd_path, mgr)

    new_sdd = cross.applyOneSDD(old_sdd, mgr, cross_args)

    save_vtr_path(mgr.vtree(), vtr_path)
    save_sdd_path(new_sdd, sdd_path)

def load_sdd(sdd_path, mgr):
    lsdd = mgr.read_sdd_file(sdd_path.encode())
    lsdd.ref()
    return lsdd

def load_manager(vtr_path):
    vtr = Vtree.from_file(vtr_path)
    mgr = SddManager.from_vtree(vtr)
    mgr.auto_gc_and_minimize_on()
    return mgr

def save_models(children, run_name):
    return [save_model(child, run_name, i) for i, child in enumerate(children)]

def save_model(mdl, run_name, num):
    dir = "ga/crossover/tmp_cross"
    path = os.path.join(dir, run_name)

    if not os.path.exists(path):
        os.makedirs(path)

    return save_sdd(mdl.sdd, path, num), save_vtr(mdl.mgr.vtree(), path, num)

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

pp = mp.Pool(n_cores)
