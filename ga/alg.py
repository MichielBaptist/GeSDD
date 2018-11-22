import numpy as np
import random

def run(params):
    selection = params['selection']
    paring = params['paring']
    cross = params['cross']
    mutation = params['mutation']
    fitness = params['fitness']
    generator = params['generator']
    
    pop_size = params['pop_size']
    data = params['data']    
    mpr = params['mutate_probability']
    n_gen = params['n_gens']
    n_select = params['n_select']
    
    population = generator.gen_n(pop_size)
    population = optimize_population(population, data)
    population_fitness = fitness_population(population, fitness, data)
    
    #logbook = logbook()
    
    for gen in range(1, n_gen + 1):
        
        # 1) Selecting the best individuals
        selected, not_selected = selection.apply(population, population_fitness, n_select)
        
        
        selected_individuals, selected_fitness = zip(*selected)
        non_selected_individuals, non_selected_fitness = zip(*not_selected)
        
        print(selected_individuals)
        print(non_selected_individuals)
        quit()
        
        # 2) Pair individuals
        pairs_for_mating = pairing.pair(selected_individuals, selected_fitness)
        
        # 3) Produce offspring
        offspring_of_selected = cross_pairs(pairs_for_mating, cross)
        
        # 4) Join the population again
        population = union_of_sets(offspring_of_selected, not_selected)
        
        # 5) Mutate the population
        population = mutate_population(population, mutation, mpr)
        
        # 6) Re-optimize the population and calculate fitness
        population = optimize_population(population, data)
        population_fitness = fitness_population(population, fitness)
        
    return population, population_fitness
        
def union_of_sets(left, right):
    return left+right
    
def cross_pairs(pairs, cross):
    return [cross(left, right) for (left, right) in pairs]
    
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
    

'''
class logbook:

    def __init__(self):
        self.book = []

    def post(it, prop, items):
        if 
        self.add_column(self.book, prop, items)
    def add_column(self, table, name, data)
        
'''
