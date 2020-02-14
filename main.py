# Custom imports
from gen.gen import random_generator, data_seeded_generator

import utils.time_utils as tut

from model.model import Model
from model.indicator_manager import indicator_manager
from model.count_manager import count_manager

from datio import IO_manager
from model_IO import model_io

from Saver import saver
import aggregators as aggr

from logic import *

import ga.selection as ga_selection
import ga.crossover.crossover as ga_crossover
import ga.mutation.mutation as ga_mutation
import ga.fitness as ga_fitness

import algorithm_multi as ga_algorithm
from algorithm_multi import logbook

# General imports
import math
import random
import pysdd
import numpy as np
import sys
import os
import argparse

from pysdd.sdd import SddManager, Vtree
from sklearn.model_selection import train_test_split
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description='''GeSDD is a genetic algorithm for learning Markov logic networks. This project was made
                                                by Michiel Batist as part of a master's thesis. The thesis text can be found in the github repository.
                                                It goes into detail of the theoretical working of GeSDD. It explains all the parameters described
                                                below. For more information about each parameter, consult the thesis text under section \'GeSDD\'.

                                                I would like to take this opportunity to thank Prof. Dr. Ir. Luc De Raedt, Prof. Dr. Jesse Davis, Dr. Jessa Bekker and Pedro Zuidberg Dos Martires.
                                                ''')
parser.add_argument('-train', required = True, help='The train data set')
parser.add_argument('-valid', required = True, help='The validation data set')
parser.add_argument('-run_folder',
                    required = False,
                    help=
                    "\n".join(["Name of the top level directory where temporary results of the run are stored.",
                               "The temporary results of a run are stored periodically. When cancelling the training",
                               "one can resume from the latest temporary results.",
                               "",
                               "Results of a particular run are saved in the folder: \<run_folder>\<run_name>\ ",
                               ""]),
                    default = "current_runs")
parser.add_argument('-run_name',
                    required = False,
                    help='''Name of the run, used for creating temporary folders inside the <run_name> folder.

                    Results of a particular run are saved in the folder: \<run_folder>\<run_name>\
                    ''' + '\n',
                    default = "default_run")
parser.add_argument('-alpha',
                    help='''Decides how much SDD compactness to be preferred over fit.''',
                    action = 'store',
                    type = float,
                    default = 5e-5
                    )
parser.add_argument('-population_size',
                    help='''The size of the population.''',
                    action='store',
                    type = int,
                    default=52)
parser.add_argument('-n_gens',
                    help='''Determines the number of generations GeSDD will be run for.''',
                    action = 'store',
                    type = int,
                    default = 30)
parser.add_argument('-mutate_p',
                    help='''Determines the probability of mutation in the population.''',
                    action = 'store',
                    type = float,
                    default = 0.8)
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

    # Generate a random seed, the seed is stored in the parameter list
    # for reproducibility.
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

    # bot: The lowest possible indicator variable for factors
    # top: The highest possible indicator variable for factors
    bot = n_vars + 1
    top = bot + args.max_nb_f

    # Initialize the indicator manager. The indicator manager will make sure of:
    # 1) The maximum allowed amount of factors in total is not exceded.
    # 2) The same factors will get the same indicator.
    # 3) Keeps track of which indicators are still available.
    # These operations are not of theoretical concern, only implementation details.
    imgr  = indicator_manager(range(bot, top))
    # Initialize the count manager. The count manager simply computes
    # all datapoints in a binary dataset such that a certain factor is satisfied.
    #
    #   S = { x | x \in \mathcal{X}, f(x) = 1}
    #
    # As computing the subset of all satisfying datapoints for a given
    # factor is computationally expensive, there is caching involved.
    # The implementation of the count manager is not of theoretical concern,
    # only implementation details. Inspect the count manager class for more details.
    cmgr = count_manager()

    # Compresses a set of binary vectors X = [x, ..., z] to a set of
    # tuples [(n, x), ..., (m, z)] such
    # that n is the amount of times x is contained in X
    #   n = count(x, X)
    #   m = count(z, X)
    # This is done for efficiency reasons
    train = cmgr.compress_data_set(train, "train")
    valid = cmgr.compress_data_set(valid, "valid")

    # Select the feature generator. If seeded, the feature generator by
    # Jan van Haaren et al. If not seeded, use the random subset feature
    # generator. For more information on these generators, inspect gen/gen.py
    if args.generator == 'seeded':
        gen = data_seeded_generator(train,  None, cmgr, top)
    elif args.generator == 'random':
        gen = random_generator(n_vars, top, cmgr)

    # Compute the empty models log likelihood on the training data.
    # This value is used in the fitness function of GeSDD. In order to give
    # more importance to relative differences in log likelihood.
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

    params['fitness'] = ga_fitness.fitness5(args.alpha, empty_ll)
    params['generator'] = gen
    params['logbook'] = logbook()
    params['pop_size'] = args.population_size
    params['train'] = train
    params['valid'] = valid
    params['mutate_probability'] = args.mutate_p
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

    # Run the algorithm, this is whare all of the work is performed.
    pop, params= ga_algorithm.run(params, cnt = (args.cnt == "yes"))

    # Find the base LL for training and validation data.
    # These are used by data aggregators.
    zero_t_ll = empty_model.LL(train, "train")
    zero_v_ll = empty_model.LL(valid, "valid")
    zero_t_fit = params['fitness'].of(empty_model, train, "train")
    zero_v_fit = params['fitness'].of(empty_model, valid, "valid")

    params['logbook'].post(0, "zero_t_ll", zero_t_ll)
    params['logbook'].post(0, "zero_v_ll", zero_v_ll)
    params['logbook'].post(0, "zero_t_fit", zero_t_fit)
    params['logbook'].post(0, "zero_v_fit", zero_v_fit)

    # Aggregators are a top level decision
    aggregators = [
        aggr.FITNESS_TRAIN,
        aggr.FITNESS_VALID,
        aggr.LL_TRAIN,
        aggr.LL_VALID,
        aggr.TIMES,
        aggr.SIZES,
        aggr.BEST_AND_SIZE,
        aggr.BEST_IND,
        aggr.NB_FACTORS,
        #aggr.LIVE_DEAD_SIZE,
        aggr.INDICATOR_PROFILE,
        aggr.BEST_MODEL,
        aggr.MODEL_EFFICIENY,
        aggr.FEATURE_SIZES
    ]

    # Finally, save the run to the "run" folder. The saver will
    # automatically create a new folder withing "run" based on date and time.
    # A run.txt file is created, alongside all the output of the aggregators.
    # The run.txt file contains all the parameters needed to reproduce the run.
    svr = saver("run")
    svr.save_run(params, logbook, aggregators)


if __name__ == "__main__":
    args = parser.parse_args()
    __main__(args)
