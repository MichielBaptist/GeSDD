import numpy as np
import random
import utils.string_utils as stru
import utils.time_utils as tut
import optimizer.multiprog_model_fitter as multiprog_fitter
import ga.crossover.crossover_application as crossover_application
import ga.mutation.mutation_application as mutation_application

from functools import reduce
from matplotlib import pyplot as plt
from itertools import chain

import time


def run(params):

    print(params)

    selection = params['selection']
    pairing = params['paring']
    cross = params['cross']
    mutation = params['mutation']
    fitness = params['fitness']
    generator = params['generator']

    pop_size = params['pop_size']
    train = params['train']
    valid = params['valid']
    mpr = params['mutate_probability']
    n_gen = params['n_gens']
    n_select = params['n_select']

    book = params['logbook']
    imgr = params['indicator_manager']

    clean_tmp_out(params['tmp_path'])

    population = generator.gen_n(pop_size)

    population = fit_population_multitprog(population, train)
    print("Here now!!!!")
    population_fitness = fitness_population(population, fitness, train)

    sizes = [model.sdd_size() for model in population]
    lls_t = [model.LL(train) for model in population]
    lls_v = [model.LL(valid) for model in population]
    validation_fitness = fitness_population(population, fitness, valid)

    book.post(0, "ll_t", lls_t)
    book.post(0, "fit_t", population_fitness)
    book.post(0, "ll_v", lls_v)
    book.post(0, "fit_v", validation_fitness)
    book.post(0, "best_ind", population[np.argmax(population_fitness)].to_string())
    book.post(0, "sizes", sizes)
    book.post(0, "best_size", sizes[np.argmax(population_fitness)])
    book.post(0, "best_ind", population[np.argmax(population_fitness)].to_string())

    t1 = time.time()
    tmr = tut.get_timer()

    tmr.start()


    print(f"Sizes: {[mdl.sdd.size() for mdl in population]}")
    print(f"Lives: {[mdl.mgr.live_size() for mdl in population]}")

    for gen in range(1, n_gen + 1):

        t1 = time.time()
        tmr.restart()

        # 1) Selecting the best individuals
        selected, not_selected = selection.apply(population, population_fitness, n_select)

        t2 = time.time()
        tmr.t("selection")

        selected_individuals, selected_fitness = selected
        non_selected_individuals, non_selected_fitness = not_selected

        t3 = time.time()
        tmr.t("misc_1")

        # 2) Pair individuals
        pairs_for_mating = pairing.pair(selected_individuals, selected_fitness)

        tmr.t("pairing")
        t4 = time.time()

        # 3) Produce offspring
        offspring_of_selected = cross_pairs(pairs_for_mating, cross)

        t5 = time.time()
        tmr.t("crossing")

        # 4) Join the population again
        #   - Non selected individuals
        #   - Selected individuals
        #   - Offspring of selected
        population = union_of_sets( offspring_of_selected,
                                    non_selected_individuals,
                                    selected_individuals
                                    )

        t6 = time.time()
        tmr.t("misc_2")

        # 5) Mutate the population
        population = mutate_population(population, mutation, mpr)

        t7 = time.time()
        tmr.t("mutation")

        # 6) Re-optimize the population and calculate fitness
        population = fit_population_multitprog(population, train)

        tmr.t("fitting")

        #for mdl in population:
        #    mdl.mgr.garbage_collect()

        population_fitness = fitness_population(population, fitness, train)

        tmr.t("find_fitness")

        population, population_fitness = kill_population(population, population_fitness, pop_size)

        t8 = time.time()
        tmr.t("killing")

        lls_t = [model.LL(train) for model in population]
        lls_v = [model.LL(valid) for model in population]

        #print(lls_v)

        validation_fitness = fitness_population(population, fitness, valid)
        sizes = [model.sdd_size() for model in population]
        nb_facts = [model.nb_factors for model in population]
        best_nb_factors = nb_facts[np.argmax(population_fitness)]

        best_index = np.argmax(population_fitness)
        best_fit_v = validation_fitness[best_index]
        best_fit_t = population_fitness[best_index]
        best_ll_t = lls_t[best_index]
        best_ll_v = lls_v[best_index]
        best_model = population[best_index].soft_clone()    # not hard clone

        t9 = time.time()
        tmr.t("info_gather")

        print(f"Sizes: {[mdl.sdd.size() for mdl in population]}")
        print(f"Lives: {[mdl.mgr.live_size() for mdl in population]}")

        for mdl in population:
            mdl.mgr.garbage_collect()

        book.post(gen, "ll_t", lls_t)
        book.post(gen, "fit_t",population_fitness)
        book.post(gen, "ll_v", lls_v)
        book.post(gen, "fit_v", validation_fitness)
        book.post(gen, "time: selection", t2- t1)
        book.post(gen, "time: nothing", t3- t2)
        book.post(gen, "time: pairing", t4- t3)
        book.post(gen, "time: crossing", t5- t4)
        book.post(gen, "time: union", t6- t5)
        book.post(gen, "time: mutation", t7- t6)
        book.post(gen, "time: fitting", t8- t7)
        book.post(gen, "time: selection", t2- t1)
        book.post(gen, "time", t9 - t1)
        book.post(gen, "time: extra", t9 - t8)
        book.post(gen, "sizes", sizes)
        book.post(gen, "best_size", sizes[np.argmax(population_fitness)])
        book.post(gen, "best_ind", population[np.argmax(population_fitness)].to_string())
        book.post(gen, "nb_factors", nb_facts)
        book.post(gen, "best_nb_factors", best_nb_factors)
        book.post(gen, "indicator_profile", imgr.profile())
        book.post(gen, "best_model", (best_model,
                                      best_model.sdd_size(),
                                      best_fit_v,
                                      best_fit_t,
                                      best_ll_v,
                                      best_ll_t))

        print(f"iteration: {gen} took {t9-t1}")
        tmr.summary(f"Summary of iteration {gen}.")

        sum_tbl = [
            ("Attr.", "Best", "Worst", "Avg", "misc"),
            ("Pop Fit (T)", max(population_fitness), min(population_fitness), np.mean(population_fitness), "/"),
            ("Pop Fit (V)", max(validation_fitness), min(validation_fitness), np.mean(validation_fitness), "/"),
            ("Pop LL (T)", max(lls_t), min(lls_t), np.mean(lls_t), "/"),
            ("Pop LL (V)", max(lls_v), min(lls_v), np.mean(lls_v), "/"),
            ("Pop size", max(sizes), min(sizes), np.mean(sizes), "/"),
        ]
        sum_str_stats = stru.pretty_print_table(sum_tbl, top_bar = True, bot_bar=True, name=f"Population statistics for gen {gen}")
        best_str_stats = stru.pretty_print_table(
            [
                ("LL (T)","LL (V)", "Fit (T)", "Fit (V)", "size"),
                (best_ll_t, best_ll_v, best_fit_t, best_fit_v, sizes[best_index])
            ],
            top_bar = True,
            bot_bar = True,
            name = "Best model statistics"
        )
        sum_str_times = tmr.summary_str(f"Summary of iteration {gen}.")



        write_to_tmp(params['tmp_path'], best_str_stats, sum_str_stats, sum_str_times)
        print(best_str_stats)

    return population, population_fitness

def clean_tmp_out(path):
    f = open(path, "w")
    f.write("")
    f.close()

def write_to_tmp(path, *str):
    with open(path, "a") as f:
        for s in str:
            f.write(s)

def kill_population(population, fitness, pop_size):
    ordered_population = np.argsort(fitness)
    top_population = ordered_population[-pop_size:]
    bot_population = ordered_population[:len(ordered_population)-pop_size]

    survivors = [population[i] for i in top_population]
    survivors_fitness = [fitness[i] for i in top_population]

    victims = [population[i] for i in bot_population]

    # Kill off the unlucky models
    for model in victims:
        model.free()

    return survivors, survivors_fitness

def fit_population_multitprog(population, data):
    return multiprog_fitter.fit_population_multitprog(population, data)

def union_of_sets(*sets):
    return reduce(lambda x, y: x + y, sets)

def cross_pairs(pairs, cross):
    return crossover_application.cross_pairs(pairs, cross)

def cross_pairs_multiprog(pairs, cross):
    return crossover_application.cross_pairs_multiprog(pairs, cross)

def optimize_population(population, data):
    [individual.fit(data) for individual in population]
    return population

def mutate_population_multiprog(population, mutation, mpr):
    return mutation_application.mutate_population_multiprog(population, mutation, mpr)

def mutate_population(population, mutation, mpr):
    return mutation_application.mutate_population(population, mutation, mpr)

# Mutatue based on probability
def mutate_wrap(individual, mutation, mpr):
    if not should_mutate(mpr):
       return individual
    else:
       return mutation.apply(individual)

def should_mutate(mpr):
    return np.random.rand() < mpr

def fitness_population(pop, fit, data):
    return [fit.of(ind, data) for ind in pop]



# --------------------------- MISC ---------------------------------------------


# Debug for info
def aggregate_and_show(times):
    # times: [[(section, time)], ........, [(section, time)]]

    uniques = set([sec for lst in times for (sec, _) in lst])

    tbl = []
    for section in uniques:
        sum = 0
        for lst in times:
            lst = dict(lst)
            if section in lst:
                sum += lst[section]
        tbl.append((section, sum))

    print(stru.pretty_print_table(tbl, top_bar=True, bot_bar=True, name="Summary of fitting."))


class logbook:

    def __init__(self):
        # Per iteration an index
        self.book = {}

    def post(self, it, prop, items):
        if it not in self.book:
            self.book[it] = {}
        self.book[it][prop] = items

    def get_prop(self, name):
        return [data[name] for (i, data) in self.book.items() if name in data]

    def get_iteration(self, it):
        return self.book[it]

    def get_point(self, it, name):
        return self.book[it][name]

    def unique_properties(self):
        unique_properties = [v for (k,v) in self.book.items()]
        unique_properties = [list(d.keys()) for d in unique_properties]
        unique_properties = set([i for l in unique_properties for i in l])
        return unique_properties

    def __contains__(self, key):
        return key in self.unique_properties()

    def __str__(self):
        n_iter = len(self.book.items())
        unique_props = self.unique_properties()

        props = [("Prop. Nb.", "Prop.")]
        props += [(i+1, p) for i, p in enumerate(unique_props)]
        props = stru.pretty_print_table(props)
        props = stru.pretty_print_table([("Properties:", props)])
        lines = [
            "Standard logbook",
            f"--> Nb. iterations: {n_iter}",
            props
        ]

        return "\n".join(lines)
