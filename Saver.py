import os
from datetime import datetime

import utils.string_utils as stru
import numpy as np

class saver:

    def __init__(self, path, display=True, run_file = "run.txt"):
        self.path = path
        self.display = display
        self.run_file = run_file

    def save_run(self, params, logbook, aggregators):

        # Make sure the folder for runs exists
        run_path = self.path

        # If it doesn't then create it
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        # Create the new run folder
        new_run_name = self.now()
        new_run_folder = os.path.join(run_path, new_run_name)

        if not os.path.exists(new_run_folder):
            os.makedirs(new_run_folder)

        # 1) Create the run configuration file
        self.create_run_file(new_run_folder, params)

        # 2) For each property that was recorded, make a file with that name
        self.dump_data(new_run_folder, params['logbook'])

        # 3) Aggregate the data using given aggregators and current save path
        for aggr in aggregators:
            aggr(params['logbook'], new_run_folder)

    def now(self):
        i = datetime.now()
        return i.strftime('%y-%m-%d||%H:%M:%S')


    def create_run_file(self, direct, params):
        # Open
        rfile = open(os.path.join(direct, self.run_file), "w")

        # Don't put data in this file
        params['train'] = None
        params['valid'] = None

        table = list(params.items())
        pretty_table = stru.pretty_print_table(table)

        print(pretty_table)

        rfile.write(pretty_table)
        rfile.close()

    def dump_data(self, direct, logbook):
        for k in logbook.unique_properties():
            self.dump_to_file(direct, k, logbook.get_prop(k))

    def dump_to_file(self, direct, name, data):

        np.savetxt(os.path.join(direct, name+".dat"), np.array(data))
