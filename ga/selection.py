import numpy as np

class selection:
    def apply(self, pop, fitness, n):
        # pop and fitness should be equal size, where each pair formed by (pop[i], fitness[i])
        # should be the fitness of individual i
        pass

# Implements basic weighted selection
# Except there is a certain regularizer
class Weighted_selection(selection):
    def __init__(self, reg):
        self.regularizer = reg

    def apply(self, pop, fitness, n):
        pop_n = len(pop)
        probs = [fi/sum(fitness) for fi in fitness]
        selected_ind = np.random.choice(range(pop_n), replace = False, p=probs, size = n)
        non_selected_ind = [i for i in range(pop_n) if i not in selected_ind]

        s_pop =  take_indices(pop, selected_ind)
        s_fit =  take_indices(fitness, selected_ind)
        ns_pop = take_indices(pop, non_selected_ind)
        ns_fit = take_indices(fitness, non_selected_ind)

        return (s_pop, s_fit), (ns_pop, ns_fit)


    def __str__(self):
        lines = [
            "Weighted selection without regularizer",
            "--> P(i) = wi / Z"
        ]
        return "\n".join(lines)

class tournament_selection(selection):
    def __init__(self, k = 3):
        self.k = k

    def apply(self, pop, fitness, n):

        ns_individuals = pop
        ns_fits = fitness
        s_individuals = []
        s_fits = []
        for i in range(n):
            # select individual
            selected_ind, selected_fit, ns_individuals, ns_fits = self.select_one(ns_individuals, ns_fits, self.k)
            # delete from remaining
            s_individuals.append(selected_ind)
            s_fits.append(selected_fit)

        return (s_individuals, s_fits), (ns_individuals, ns_fits)

    def select_one(self, pop, fits, k):
        if len(pop) != len(fits):
            print("Pop length was not fit length!")
            #print(pop)
            #print(fits)

        # Select your tournament
        indices = list(range(len(pop)))
        selected_k = list(np.random.choice(indices, size=k, replace = False))

        #print(f"selected_k (indices): {selected_k}")

        selected_fits = take_indices(fits, selected_k)

        #print(f"selected fits: {selected_fits}")

        ws = [f/sum(selected_fits) for f in selected_fits]

        selected_ind = np.random.choice(list(range(k)), p = ws)
        #print(f"selected ind (k): {selected_ind}")
        selected_ind = selected_k[selected_ind]
        #print(f"selected ind (index): {selected_ind}")

        ns_pop = [p for i, p in enumerate(pop) if i != selected_ind]
        ns_fit = [f for i, f in enumerate(fits) if i != selected_ind]

        #print(f"ns_pop: {ns_pop}")
        #print(f"ns_fit: {ns_fit}")

        return pop[selected_ind], fits[selected_ind], ns_pop, ns_fit

class softmax_selection(selection):
    def __init__(self, reg):
        self.regularizer = reg

    def apply(self, pop, fitness, n):
        pop_n = len(pop)
        probs = [math.exp(fi) for fi in fitness]
        probs = [fi / sum(probs) for fi in probs]
        selected_ind = np.random.choice(range(pop_n), replace = False, p=probs, size = n)
        non_selected_ind = [i for i in range(pop_n) if i not in selected_ind]

        s_pop =  take_indices(pop, selected_ind)
        s_fit =  take_indices(fitness, selected_ind)
        ns_pop = take_indices(pop, non_selected_ind)
        ns_fit = take_indices(fitness, non_selected_ind)

        return (s_pop, s_fit), (ns_pop, ns_fit)


    def __str__(self):
        lines = [
            "Weighted selection without regularizer",
            "--> P(i) = wi / Z"
        ]
        return "\n".join(lines)


class pairing:
    def pair(self, pop, fit):
        zipped = zip(pop, fit)
        sorted_pop = sorted(zipped, key = lambda x : x[1], reverse=False)
        print([(mdl.sdd_size(), ft) for (mdl, ft) in sorted_pop])

        pairs = []
        for i in range(0, len(sorted_pop), 2):
            pairs.append((sorted_pop[i][0], sorted_pop[i+1][0]))

        return pairs

    def __str__(self):
        return "Standard ordered pairing"

# ---------------------------------- MISC FUNC. --------------------------------
def take_indices(arr, ind):
    return [arr[i] for i in ind]
