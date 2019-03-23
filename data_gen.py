import numpy as np

def save(path, dat, delim):
    np.savetxt(path, dat, fmt = '%i', delimiter=delim)

def gen(model, n_worlds):

    # Horribly inefficient method of generating
    # a data set using a given model.

    # This function loops all possible worlds and
    # finds their prob. and then samples from this.

    # Domain size?
    domain_size = model.domain_size

    # Get all configs (vary bad)!
    all_worlds = find_all_worlds(domain_size)

    # Find the probability of all worlds
    probs = [model.world_probability(world) for world in all_worlds]

    # Sample n_worlds from this distirubtion
    choices = np.random.choice(range(len(all_worlds)), size = n_worlds, replace = True, p=probs)
    #TODO: Clean this up somehow
    samples = tuple([tuple(all_worlds[i]) for i in choices])

    return samples

def find_all_worlds(domain_size):
    if domain_size == 1:
        return [[True], [False]]
    else:
        worlds_rest = find_all_worlds(domain_size-1)
        worlds_all = []
        for world in worlds_rest:
            t = list(world)
            t.append(True)
            f = list(world)
            f.append(False)

            worlds_all.append(t)
            worlds_all.append(f)
        return worlds_all
