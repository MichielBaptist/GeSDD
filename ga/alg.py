import numpy as np
import random

from matplotlib import pyplot as plt
from itertools import chain

def run(params):
    selection = params['selection']
    pairing = params['paring']
    cross = params['cross']
    mutation = params['mutation']
    fitness = params['fitness']
    generator = params['generator']
    
    pop_size = params['pop_size']
    data = params['data']    
    mpr = params['mutate_probability']
    n_gen = params['n_gens']
    n_select = params['n_select']
    
    book = params['logbook']
    
    population = generator.gen_n(pop_size)
    population = optimize_population(population, data)
    population_fitness = fitness_population(population, fitness, data)
    
    lls = [model.LL(data) for model in population]
    
    book.post(0, "ll", lls)
    book.post(0, "fit", population_fitness)
    
    for gen in range(1, n_gen + 1):
    
        print(f"iteration: {gen}")
        
        # 1) Selecting the best individuals
        selected, not_selected = selection.apply(population, population_fitness, n_select)        
        
        selected_individuals, selected_fitness = selected
        non_selected_individuals, non_selected_fitness = not_selected
        
        # 2) Pair individuals
        pairs_for_mating = pairing.pair(selected_individuals, selected_fitness)
        
        # 3) Produce offspring
        offspring_of_selected = cross_pairs(pairs_for_mating, cross)
                
        # 4) Join the population again
        population = union_of_sets(offspring_of_selected, non_selected_individuals)
        
        # 5) Mutate the population
        population = mutate_population(population, mutation, mpr)
        
        # 6) Re-optimize the population and calculate fitness
        population = optimize_population(population, data)
        population_fitness = fitness_population(population, fitness, data)
        
        
        lls = [model.LL(data) for model in population]
        
        book.post(gen, "ll", lls)
        book.post(gen, "fit",population_fitness)
        
    return population, population_fitness
        
def union_of_sets(left, right):
    return left+right
    
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
        all_data = [data[name] for (i, data) in self.book.items()]
        return all_data
        
    def get_iteration(self, it):
        return self.book[it]
        
    def get_piont(self, it, name):
        return self.book[it][name]
       
    
    
    
        
