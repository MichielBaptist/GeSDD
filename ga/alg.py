import numpy as np
import random
import utils.string_utils as stru

from functools import reduce
from matplotlib import pyplot as plt
from itertools import chain

import time

def run(params):
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
    mgr = params['manager']
    imgr = params['indicator_manager']

    population = generator.gen_n(pop_size)
    population = optimize_population(population, train)
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

    for gen in range(1, n_gen + 1):

        print(f"iteration: {gen}")

        t1 = time.time()

        # 1) Selecting the best individuals
        selected, not_selected = selection.apply(population, population_fitness, n_select)

        t2 = time.time()

        selected_individuals, selected_fitness = selected
        non_selected_individuals, non_selected_fitness = not_selected

        t3 = time.time()

        # 2) Pair individuals
        pairs_for_mating = pairing.pair(selected_individuals, selected_fitness)

        t4 = time.time()

        # 3) Produce offspring
        offspring_of_selected = cross_pairs(pairs_for_mating, cross)

        t5 = time.time()

        # 4) Join the population again
        #   - Non selected individuals
        #   - Selected individuals
        #   - Offspring of selected
        population = union_of_sets( offspring_of_selected,
                                    non_selected_individuals,
                                    selected_individuals
                                    )

        t6 = time.time()

        # 5) Mutate the population
        population = mutate_population(population, mutation, mpr)

        t7 = time.time()

        if gen % 2 == 0 and False:

            print("Starting to minimize!")

            for model in population:
                model.ref()

            mgr.minimize()

            for model in population:
                model.deref()

            print("done minimizing")



        # 6) Re-optimize the population and calculate fitness
        population = optimize_population(population, train)

        for m in population:
            if m.LL(train) > 0:
                print(m.to_string())

        population_fitness = fitness_population(population, fitness, train)

        population, population_fitness = kill_population(population, population_fitness, pop_size)

        t8 = time.time()

        lls_t = [model.LL(train) for model in population]
        lls_v = [model.LL(valid) for model in population]
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
        book.post(gen, "live_size", mgr.live_size())
        book.post(gen, "dead_size", mgr.dead_size())
        book.post(gen, "indicator_profile", imgr.profile())
        book.post(gen, "best_model", (best_model,
                                      best_fit_v,
                                      best_fit_t,
                                      best_ll_v,
                                      best_ll_t))

    return population, population_fitness

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

def union_of_sets(*sets):
    return reduce(lambda x, y: x + y, sets)

def cross_pairs(pairs, cross):
    children = [cross.apply(left, right) for (left, right) in pairs]
    return list(chain(*children))

def optimize_population(population, data):
    return [individual.fit(data) for individual in population]

def mutate_population(population, mutation, mpr):
    return [mutate_wrap(individual, mutation, mpr) for individual in population]

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
