import os
import utils.time_utils as tut
import utils.string_utils as stru
import numpy as np
import math
import shutil

from functools import reduce
import multiprocessing as mp
from pysdd.sdd import Vtree, SddManager
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# These are parameters not really globals...
n_cores = 6

default_tmp_dir = "optimizer/tmp"
default_sdd_name = "tmp_sdd_{}.sdd"
default_vtree_name = "tmp_vtree_{}.vtr"

default_maxit = 9
default_acc = 5e-2

# This is a global but for debugging:

def hist(times):
    plt.hist(times, bins=50)
    plt.show()

# Gets a population and returns the population such that
# every model is optimized according to the data.
def fit_population_multitprog(population, data, set_id, run_name):
    t = tut.get_timer()

    tmp_run_dir = os.path.join(default_tmp_dir, run_name)

    t.start()

    # If the population is empty just return nothing.
    if len(population) == 0:
        return []

    # 1) Count all the factors in the count manager such that all counts of
    #    the rules are cached. This potentially reduces work a lot.
    #    This method will return a list of counts for each model in the pop.
    #    This is done sequentially, but takes almost no time.
    data_counts = count_population_factors(population, data, set_id)
    t.t("counting")

    # 2) Obtain a mapping of all trainable indicators to initial weights:
    #    Should return: [[(i, w), ..., (i, w)] ......... [(i, w), ..., (i, w)]]
    initial_weights = find_initial_weights(population)

    # 3) Save all SDDs to file so that it can be re-loaded in another thread.
    tmp_vtrees = tmp_save_vtree(population, tmp_run_dir)
    tmp_sdds = tmp_save_sdds(population, tmp_run_dir)

    t.t("Saving vtree, sdd")
    # 4) Loop over all models and start a new thread for each. This happens
    #    using multiprocessing. Note the results of training are still in the
    #    correct ordering. After which all training results get applied.
    train_results = train_all_models(population, tmp_sdds, tmp_vtrees, initial_weights, data_counts)

    t.t("Train")
    population = apply_train_results(population, train_results)
    t.t("Apply")

    # Summary of the entire parallel part.
    t.summary("Summary of parallell fitting.")

    times = [res["total_time"] for res in train_results]
    show_parallell_summary(times, t)

    return population

def show_parallell_summary(times, t):
    tbl = [
        ("Measure", "Value", "Expl."),
        ("Max time (Ma)", max(times), "/"),
        ("Min time (Mi)", min(times), "/"),
        ("Mean time (M)", np.mean(times), "/"),
        ("Summed time (S)", sum(times), "/"),
        ("Actual time (A)", t.total(), "/"),
        ("Gain Ratio", t.total()/sum(times), "(A/S). Percentage of work done."),
        ("Efficiency ratio", max(times)/t.total(), "(Ma/A)."),
        ("Standard deviation", np.std(times), "Spread of time")
    ]
    print(stru.pretty_print_table(tbl, top_bar = True, bot_bar = True, name="Time statistics of jobs."))


def apply_train_results(population, train_results):
    # population: [model] collection of models
    # train_results: [result] collection of results.
    # The result of populaion[i] is train_results[i]
    """times = []
    for res in train_results:
        times.append(res["train_time"])
    hist(times)"""

    return [apply_train_result(model, result) for (model, result) in zip(population, train_results)]

def apply_train_result(model, result):
    model.set_factor_weights(result["w"])
    return model

# ----------------- Training models -----------------
def train_all_models(population, sdd_paths, vtree_paths, initial_weights, data_counts):
    # sdd_paths: [path_to_an_SDD]
    # vtree: [path_to_a_vtree]
    # initial_weights: [ [(i, w)] ... ] list of indicator weights pairs per model
    # data_counts: [ [(i, c)] ... ] list of counts per factor per model

    # Extract extra information needed and put it in args later.
    domain_size = extract_domain_size(population)
    n_all_literals = extract_total_n_literals(population)

    # Merge all arguments to make it easier
    arg_names = [
            "sdd_path",
            "vtree_path",
            "init_weights",
            "data_counts",
            "domain_size",
            "n_all_literals"
    ]
    model_args = zip(
                     sdd_paths,
                     vtree_paths,
                     initial_weights,
                     data_counts,
                     [domain_size]*len(sdd_paths),
                     [n_all_literals]*len(sdd_paths)
    )
    model_args = [dict(zip(arg_names, model)) for model in model_args]

    # This is the most important part of the
    # function. It creates a pool of "threads"
    # and maps training the models to these "threads".
    #mp.set_start_method("forkserver")

    results = pp.map(train_model, model_args)

    return results

def train_model(args):
    # args: {par:val}
    #   - sdd_path: path to the sdd to load in
    #   - vtree_path: path to the vtree
    #   - init_weights: initial weights of the model. This also contains all the
    #                   trainable indicators the model contains. All other
    #                   indicator weights are always set to O (log)/1 (normal).
    #   - data_counts: how often the data says the indicator is true. For literal
    #                 indicators this will just count how often the literal is true
    #                 in the data. For non literal indicators it will count how
    #                 often the corresponding rule is true in the data.
    #   - domain_size: which literals are normal literals (not indicators)
    #   - n_all_literals: how many literals are there in total? (including indicators)

    # This is the most important method. It is always run in parallel to other
    # models. It does not have access to shared data sctructure.

    #1) Load in the manager and SDD
    vtree = load_vtree(args["vtree_path"])
    mgr = manager_from_vtree(vtree)
    sdd = mgr.read_sdd_file(args["sdd_path"].encode())

    # 2) Actually perform the training
    return maximize_objective(sdd, args)

def maximize_objective(sdd, args, acc=None, maxit=None, timer=None):
    # SDD: the relevant SDD
    # Args: All arguments required to run the maximisation.
    #       See above for more info.
    # acc: How accurate should the estimate be?
    # maxit: How many iterations before done?

    if timer == None:
        timer = tut.get_timer()

    if maxit == None:
        maxit = default_maxit

    if acc == None:
        acc = default_acc

    timer.start()

    # 1) First get the objective function and gradient function.
    objective = objective_function
    gradient  = gradient_function

    # 2) Get additional informtion for the optmization.
    trainable_literals = [l for (l, _) in args["init_weights"]]
    literal_counts = [c for (_, c) in args["data_counts"]]

    # 3) Construct the dictionary which contains the parameters
    #    in order to evaluate the function.
    fn_args = {
        "sdd":sdd,
        "trainable_literals": trainable_literals,
        "literal_counts": literal_counts,
        "domain_size": args["domain_size"],
        "n_all_literals": args["n_all_literals"],
        "timer":timer
    }
    timer.t("Misc_train")
    
    # 4) Do the actual minimization.
    minimization_result = minimize(
        objective,
        args=(fn_args,),
        x0 = [init_w for (_, init_w) in args["init_weights"]],
        jac = None,
        method="BFGS",
        options={
            #'gtol': acc,
            'disp': False,
            'maxiter': maxit
        }
    )
    timer.t("Main_train")
    # 5) Extract the results and place it in a dict.
    weights = minimization_result.x
    LL = minimization_result.fun

    timer.t("train_time")

    timer.summary()

    return {
        "w":weights,
        "LL":LL,
        "total_time":timer.total()
    }

def gradient_function(weights, args):
    sdd = args["sdd"]
    literals = args["trainable_literals"]
    counts = args["literal_counts"]

    domain_size = args["domain_size"]
    n_all_literals = args["n_all_literals"]

    wmc = WMC_with_weights(sdd, weights, literals, domain_size, n_all_literals)
    wmc.propagate()

    lit_prs = [math.exp(wmc.literal_pr(l)) for l in literals]

    grads = [wmc.literal_derivative(l) for l in literals]
    gradients = [expected - actual for (actual, expected) in zip(counts, lit_prs)]

    #print(grads)
    #print(gradients)

    return gradients

def objective_function(weights, args):
    # weights: weights of the literals that are learnable
    #           - Learnable literals include the domain and all indicators for
    #             the rules.
    # args: Not the same as previous args.

    timer = args["timer"]

    timer.start()

    # Get the SDD
    sdd = args["sdd"]

    # The names of the literals for the weights given in weights.
    literals = args["trainable_literals"]

    # The counts of the literals for the weights given in weights
    #   counts: [count]
    counts = args["literal_counts"]

    # What is the domain size of model in the sdd?
    # What is the size of the manager (including all literals reserved for
    # indicators)
    domain_size = args["domain_size"]
    n_all_literals = args["n_all_literals"]

    timer.t("--unpack arguments")

    # LL = sum(count(f_i) * w_i) - ln(Z)
    # LL = SOWC - LN_Z
    # LL: Log-Likelihood
    # SOWC: Sum Of Weighted Counts
    # LN_Z: Log of partition Z
    wmc = WMC_with_weights(sdd, weights, literals, domain_size, n_all_literals)

    timer.t("--Setting literal weights")

    LN_Z = wmc.propagate()

    timer.t("--Propagation")

    SOWC = sum([w * c for (w,c) in zip(weights, counts)])
    LL = SOWC - LN_Z
    #grads = [math.exp(wmc.literal_pr(l)) - c for (l,c) in zip(literals, counts)]
    #print(f"{LL} --> {grads[0]}")

    timer.t("--Calculate result")
    return LL * -1

def WMC_with_weights(sdd, weights, literals, domain_size, n_all_literals):
    wmc = sdd.wmc(True)

    # The irrelevant literals may contain relevant literals. However this is
    # not a problem as we set the weights of the relevant literals after the
    # irrelevant ones.
    irrelevant_indicators = range(1, n_all_literals+1)

    # 1) Set all irrelevant literals to zero weight
    for i in irrelevant_indicators:
        #print(f"Setting {i} and {-i} to 0")
        wmc.set_literal_weight(i, wmc.zero_weight)
        #wmc.set_literal_weight(-i,wmc.zero_weight)

    # Sanity check: The length of the weights and literals must be the same.
    if len(literals) != len(weights):
        print("""Amount of trainable literals is not the same as the
                 amount of weights given for the partition.""")

    # 2) set the weights of the relevant literals to their weights
    for (l, w) in zip(literals, weights):
        #print(f"Setting {l} to {w}")
        wmc.set_literal_weight(l, w)

    return wmc

def partition_with_weights(sdd, weights, literals, domain_size, n_all_literals):
    # This method will compute a partition given weights of literals
    # and some additional information.

    wmc = WMC_with_weights(sdd, weights, literals, domain_size, n_all_literals)

    Z = wmc.propagate()

    #print(f"-----{[math.exp(wmc.literal_pr(l)) for l in literals]}")

    return Z

def extract_domain_size(population):
    domain_sizes = [m.domain_size for m in population]

    if not homogenous_list(domain_sizes):
        print("Warning: not all domain sizes are the same!!")

    return domain_sizes[0]

def extract_total_n_literals(population):
    total_sizes = [m.mgr.var_count() for m in population]

    if not homogenous_list(total_sizes):
        print("Warning: not all total sizes are the same!!")

    return total_sizes[0]

def load_sdd(mgr, sdd_path):
    print(f"Loading {sdd_path}")
    return mgr.read_sdd_file(sdd_path.encode())

def manager_from_vtree(vtree):
    return SddManager.from_vtree(vtree)

def load_vtree(vtree_path):
    return Vtree.from_file(vtree_path)

# -------------------- Saving functionality ----------------

def tmp_save_vtree(population, dir):
    # population:[model]
    # returns: path to the vtree file

    vtrees = extract_vtrees(population)
    path_to_vtrees = save_vtrees(vtrees, dir)

    return path_to_vtrees

def save_vtrees(vtrees, dir, path = None, vtr_name = None):
    # vtrees: [vtree]
    return [save_vtree(number, vtree, tmp_path = dir) for number,vtree in enumerate(vtrees)]

def save_vtree(number, vtree, tmp_path = None, vtr_name = None):
    # number: number of the vtree
    # vtree: vtree to save to tmp

    if tmp_path == None:
        tmp_path = default_tmp_dir
    if vtr_name == None:
        vtr_name = default_vtree_name.format(number)

    vtree_file = os.path.join(tmp_path, vtr_name)
    create_path_folders(vtree_file)

    vtree.save(vtree_file.encode())

    return vtree_file

def tmp_save_sdds(population, dir):
    # population: [model]
    # returns: path of each saved sdd
    # This method saves a list of sdds to a temporary dirself.
    # This so seperate threads are able to load them.

    return [tmp_save_sdd(number, model, dir=dir) for number, model in enumerate(population)]

def tmp_save_sdd(number, model, dir=None, name=None):
    # model: model
    # returns: a filepath containing the sdd

    # revert to default values
    if dir == None:
        dir = default_tmp_dir
    if name == None:
        name = default_sdd_name.format(number)

    # make path and add folders if they don't exist yet.
    sdd_file = os.path.join(dir, name)
    create_path_folders(sdd_file)

    sdd = model.sdd

    sdd.save(sdd_file.encode())

    return sdd_file

def create_path_folders(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# -------------- misc functionality ------------------

def extract_vtrees(population):
    # population:[model]
    return [model.mgr.vtree() for model in population]

def homogenous_list(lst):
    if len(lst)==0:
        return True

    head = lst[0]

    return reduce(lambda x,y: x and y, [el == head for el in lst])

def find_initial_weights(population):
    # population:[model]
    # returns: [[(iw)]...[(iw)]]
    return [model.get_iw() for model in population]

def count_population_factors(population, data, set_id):
    # population: [model...model]
    # returns [[(indicator, count)], ..., [(indicator,count)]]

    # First extract the count manager from the population. This should
    # be the same for every model in population.
    count_manager = extract_count_manager(population)

    # For every model extract the counts
    return [count_model_factors(model, count_manager, data, set_id) for model in population]

def extract_count_manager(population):
    # population: [model]
    # returns: count manager of the entire population
    cmgrs = [model.count_manager for model in population]

    compare = cmgrs[0]
    for c in cmgrs:
        if c != compare:
            print("Found a different count manager!!!")

    return compare

def count_model_factors(model, count_manager, data, set_id):
    # Expects a model and returns a list of indicator-count pairs
    # model: model
    # returns: [(indicator, count)]

    # First extract the count manager from the population. This should
    # be the same for every model in population.
    # 1) get (f, w, i)
    fwi = model.get_fwi()

    return [(i, count_manager.count_factor(f, data, set_id)) for (f, w, i) in fwi]

def clean_tmp_dir(run_name):
    run_dir = os.path.join(default_tmp_dir, run_name)
    shutil.rmtree(run_dir)

pp = mp.Pool(n_cores)
