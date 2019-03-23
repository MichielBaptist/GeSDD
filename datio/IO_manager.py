import numpy as np
import time
import os

def list_to_tuples_rec(list):
    return tuple(map(lambda x : tuple(x), list))

def read_from_csv_num(path, sep, verbose=True):
    start = time.time()

    fi = open(path)

    worlds = [line.split("|") for line in fi]
    worlds = [[to_boolean_array(w, sep)]*int(n) for (n, w) in worlds]
    worlds = [world for group in worlds for world in group]
    n_worlds, n_vars = np.array(worlds).shape
    #worlds = tuple([tuple(world) for world in worlds])

    print(np.shape(worlds))

    end = time.time()

    if verbose:
        notify_loaded(path, n_worlds, n_vars, end-start)

    return worlds, n_worlds, n_vars

def read_from_csv(path_to_csv, sep, verbose=True):

    start = time.time()

    csv_file = open(path_to_csv)
    worlds = [tuple(to_boolean_array(world_str, sep)) for world_str in csv_file]
    n_worlds, n_vars = np.array(worlds).shape

    end = time.time()

    if verbose:
        notify_loaded(path_to_csv, n_worlds, n_vars, (end- start))


    return worlds, n_worlds, n_vars

def to_boolean_array(world_str, sep):
    return [str_to_bool(var) for var in world_str.split(sep)]

def str_to_bool(string):
    return string.strip() == "1"

def notify_loaded(path, n_worlds, n_vars, duration):
    print("--Done loading data--")
    print(f"----> Path:  {path}")
    print(f"----> Number:{n_worlds}")
    print(f"----> Vars:  {n_vars}")
    print(f"----> Took:  {duration}")

def read_from_names_folder(folder):

    start = time.time()

    names_file = open(get_file_with_extension(folder, ".names"))
    data_file = open(get_file_with_extension(folder, ".data"))

    structure = get_data_structure(names_file)
    data = get_data_with_struct(data_file, structure)

    n_worlds = len(data)
    n_vars = structure.nb_variables()

    finish = time.time()

    notify_loaded(folder, n_worlds, n_vars, (finish - start))

    return data, structure, n_worlds, n_vars

def get_data_with_struct(data_file, data_structure):
    nb_var = data_structure.nb_variables()
    nb_attr = data_structure.nb_attributes()

    worlds = []
    for world in data_file:

        world_values = world.split(",")
        world_values = [w.strip(" \n.") for w in world_values]
        world_boolean = [False]*nb_var
        for i, val in enumerate(world_values):
            variable = data_structure.variable_of(i, val)
            world_boolean[variable-1] = True
        worlds.append(world_boolean)

    return list_to_tuples_rec(worlds)


def get_file_with_extension(folder, extension):
    all_files = [f for f in os.listdir(folder) if f.endswith(extension)]
    all_files = list(map(lambda x: os.path.join(folder, x), all_files))

    if len(all_files) > 1:
        print(f"multiple files with extension {extension} found: {all_files}")

    return all_files[0]

def get_data_structure(names_file):
    lines = names_file.readlines()
    lines = list(filter(lambda x : x != "\n", lines))
    lines = list(filter(lambda x : len(x) > 0, lines))
    lines = list(map(lambda x : x.strip(' .\n'), lines))
    lines = list(filter(lambda x : x[0] != '|', lines))

    class_line = lines[0]           # Always the first line according to c4.5
    attribute_lines = lines[1:]     # The rest are the attributes

    all_attributes = []

    classes = class_line.split(",")
    classes = list(map(lambda x : x.strip(), classes))
    class_attr = attribute("classes", classes)

    for attribute_line in attribute_lines:
        attribute_line = attribute_line.split(":")
        name = attribute_line[0]
        values = attribute_line[1].split(",")
        values = list(map(lambda x : x.strip(), values))
        attr = attribute(name, values)
        all_attributes.append(attr)


    all_attributes.append(class_attr)

    data_structure = attribute_list(all_attributes)
    return data_structure

class attribute_list:
    def __init__(self, attributes):
        self.attributes = attributes
        self.set_attribute_bases(self.attributes)
        self.set_attribute_mappings(self.attributes)

    def variable_of(self, index, name):
        return self.get_attribute(index).variable_of(name)

    def __str__(self):
        return "\n".join([attr.__str__() for attr in self.attributes])

    def set_attribute_bases(self, attributes):
        base = 0
        for attr in attributes:
            attr.set_base(base)
            base += attr.nb_values()

    def set_attribute_mappings(self, attributes):
        for attr in attributes:
            attr.set_mapping(attr.base, attr.values)


    def get_attribute(self, index):
        return self.attributes[index]

    def nb_variables(self):
        return sum([attr.nb_values() for attr in self.attributes])

    def nb_attributes(self):
        return len(self.attributes)

class attribute:
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.set_base(0)
        self.set_mapping(self.base, values)

    def set_base(self, base):
        self.base = base

    def set_mapping(self, base, values):
        mapping = {}
        variable = 1
        for value in values:
            mapping[value] = self.base + variable
            variable += 1

        self.mapping = mapping

    def variable_of(self, name):
        return self.mapping[name]

    def get_values(self):
        return self.values

    def nb_values(self):
        return len(self.values)

    def __str__(self):
        return f"{self.name}: {self.mapping}"

    def __repr__(self):
        return self.__str__()
