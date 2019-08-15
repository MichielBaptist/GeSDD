
import time
import os
import pickle
import shutil

from pysdd.sdd import SddManager, Vtree

class model_io:


    def __init__(
                self,
                dir,
                name,
                ):
        self.args_name = "run_args.pickle"
        self.all_models_dir = "models"
        self.one_model_dir_name = "mdl_{:0>3d}"
        self.one_model_dir_name_= "mdl_"
        self.common_name = "common"

        self.cmgr_name = "cmgr.pickle"
        self.imgr_name = "imge.pickle"

        self.vtree_name = "vtr.vtree"
        self.sdd_name = "sdd.sdd"
        self.model_name = "mdl.pickle"

        self.best_model_dir_name = "best_model"
        self.tmp_out_name = "tmp_out"

        self.dir = dir
        self.name = name

        # current_runs/run1/
        self.run_dir = os.path.join(dir, name)
        self.ensure(self.run_dir)

        # current_runs/run1/run_args.pickle
        self.args_pickle_path = os.path.join(self.run_dir, self.args_name)

        # current_runs/run1/models
        self.models_dir = os.path.join(self.run_dir, self.all_models_dir)
        self.ensure(self.models_dir)

        # current_runs/run1/models/common
        self.commons_dir = os.path.join(self.models_dir, self.common_name)

        # current_runs/run1/models/mdl_{}
        self.one_model_dir = os.path.join(self.models_dir, self.one_model_dir_name)

        # current_runs/run1/tmp_out
        self.tmp_out_file = os.path.join(self.run_dir, self.tmp_out_name)

        # current_runs/run1/best_model
        self.best_model_dir = os.path.join(self.run_dir, self.best_model_dir_name)

    def ensure(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_args(self, args, verbose = True):
        s = time.time()

        self.pickle_object(args, self.args_pickle_path)

        if verbose:
            print(f"Dumping to pickle took {time.time() - s}")

    def load_args(self, verbose = True):
        s = time.time()

        pckl = self.load_pickle(self.args_pickle_path)

        if verbose:
            print(f"Loading the pickle took {time.time() - s}")

        return pckl

    def load_manager(self, vtr_path):
        vtr = Vtree.from_file(vtr_path)
        mgr= SddManager.from_vtree(vtr)
        mgr.auto_gc_and_minimize_on()

        return mgr

    def load_sdd(self, mgr, sdd_path):
        return mgr.read_sdd_file(sdd_path.encode())

    def load_models(self, verbose = True):
        s = time.time()

        model_dirs = os.listdir(self.models_dir)
        model_dirs = [dir for dir in model_dirs if self.one_model_dir_name_ in dir]
        model_dirs = sorted(model_dirs)
        model_dirs = [os.path.join(self.models_dir, model_dir) for model_dir in model_dirs]
        models = [self.load_model(model_dir) for model_dir in model_dirs]

        cmgr, imgr = self.load_common_objects()

        for mdl in models:
            mdl.indicator_manager = imgr
            mdl.count_manager = cmgr

        if verbose:
            print(f"Loading {len(models)} models took {time.time() - s}s")

        return models

    def load_model(self, model_dir):
        mdl_path = os.path.join(model_dir, self.model_name)
        sdd_path = os.path.join(model_dir, self.sdd_name)
        vtr_path = os.path.join(model_dir, self.vtree_name)
        cmn_path = self.commons_dir

        mdl = self.load_pickle(mdl_path)

        mgr = self.load_manager(vtr_path)
        sdd = self.load_sdd(mgr, sdd_path)

        mdl.sdd = sdd
        mdl.mgr = mgr

        mdl.sdd.ref()

        return mdl

    def load_best_model(self):

        pth = self.best_model_dir

        return self.load_model(pth)


    def save_models(self, models, verbose = True):

        if len(models) == 0:
            print("No models given to save!")
            return

        s = time.time()

        # The count manager and indicator manager is
        # a common datatype and should be pickled separately
        count_managers = [mdl.count_manager for mdl in models]
        indic_managers = [mdl.indicator_manager for mdl in models]

        if not self.all_equal(count_managers):
            print("Not all count managers are the same object!")
        if not self.all_equal(indic_managers):
            print("Not all indicator managers are the same object!")

        cmgr = count_managers[0]
        imgr = indic_managers[0]

        self.save_common_objects(cmgr, imgr)

        # Save all models
        for i, model in enumerate(models):
            self.save_model(i, model)

        if verbose:
            print(f"Saving {len(models)} models took {time.time() - s}s")

    def save_model(self, number, model):
        dir = self.one_model_dir.format(number)

        # Ensure the models directory exists
        self.ensure(dir)

        pickle_path = os.path.join(dir, self.model_name)
        vtr_path = os.path.join(dir, self.vtree_name)
        sdd_path = os.path.join(dir, self.sdd_name)

        sdd = model.sdd
        vtr = model.mgr.vtree()

        sdd.save(sdd_path.encode())
        vtr.save(vtr_path.encode())

        model.sdd = None
        model.mgr = None
        model.count_manager = None

        self.pickle_object(model, pickle_path)

    def save_best_model(self, model):
        dir = self.best_model_dir

        self.ensure(dir)

        pickle_path = os.path.join(dir, self.model_name)
        vtr_path = os.path.join(dir, self.vtree_name)
        sdd_path = os.path.join(dir, self.sdd_name)

        sdd = model.sdd
        mgr = model.mgr
        vtr = mgr.vtree()

        sdd.save(sdd_path.encode())
        vtr.save(vtr_path.encode())

        model.sdd = None
        model.mgr = None
        cmgr = model.count_manager
        model.count_manager = None

        self.pickle_object(model, pickle_path)

        model.sdd = sdd
        model.mgr = mgr
        model.count_manager = cmgr

    def load_common_objects(self):
        t = time.time()
        cmgr_path = os.path.join(self.commons_dir, self.cmgr_name)
        imgr_path = os.path.join(self.commons_dir, self.imgr_name)

        p =  self.load_pickle(cmgr_path), self.load_pickle(imgr_path)
        print(f"Loading common obbjects took: {time.time() - t}")

        return p

    def save_common_objects(self, cmgr, imgr):
        common_dir = self.commons_dir

        self.ensure(common_dir)

        cmgr_path = os.path.join(common_dir, self.cmgr_name)
        imgr_path = os.path.join(common_dir, self.imgr_name)

        self.pickle_object(cmgr, cmgr_path)
        self.pickle_object(imgr, imgr_path)

    def pickle_object(self, obj, path):

        f = open(path, "wb")
        pickle.dump(obj, f)
        f.close()

    def load_pickle(self, path):

        f = open(path, "rb")
        obj = pickle.load(f)
        f.close()

        return obj

    def all_equal(self, lst):
        if len(lst) <= 1:
            return True

        return len(lst) == sum([e == lst[0] for e in lst])

    def write_to_tmp(self, *str):
        with open(self.tmp_out_file, "a") as f:
            for s in str:
                f.write(s)

    def clean_tmp_out(self):
        f = open(self.tmp_out_file, "w")
        f.write("")
        f.close()

    def clean_models_dir(self):
        dir = self.models_dir
        for fn in os.listdir(dir):
            path = os.path.join(dir, fn)
            print(f"Removing {path} from {dir}")
            shutil.rmtree(path)

    def show(self):
        for i in self.__dict__.items():
            print(i)
