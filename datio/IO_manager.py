import numpy as np
import time

def read_from_csv(path_to_csv, sep, verbose=False):
    
    start = time.time()
    
    csv_file = open(path_to_csv)
    worlds = [to_boolean_array(world_str, sep) for world_str in csv_file]
    n_worlds, n_vars = np.array(worlds).shape
    
    end = time.time()
    
    if verbose:
        notify_loaded(path_to_csv, n_worlds, n_vars, (end- start))
    
    
    return worlds, n_worlds, n_vars
    
def to_boolean_array(world_str, sep):
    return [str_to_bool(var) for var in world_str.split(sep)]
    
def str_to_bool(string):
    return string == "1"
    
def notify_loaded(path, n_worlds, n_vars, duration):
    print("--Done loading data--")
    print(f"----> Path:  {path}")
    print(f"----> Number:{n_worlds}")
    print(f"----> Vars:  {n_vars}")
    print(f"----> Took:  {duration}")
