import numpy as np
import utils.string_utils as stru
import os

from matplotlib import pyplot as plt


def FEATURE_SIZES(log, save_path):
    best_feature_sz = log.get_prop("best_feature_size")
    feature_sz = log.get_prop("feature_sizes")

    avg_size_pop = [np.mean([np.mean(mdl) for mdl in gen]) for gen in feature_sz]
    avg_size_best = [np.mean(gen) for gen in best_feature_sz]

    plt.plot(range(len(avg_size_pop)), avg_size_pop, label ="Average feature size pop")
    plt.plot(range(len(best_feature_sz)), avg_size_best, label="average feature size best")

    plt.xlabel("generation")
    plt.ylabel("Average feature length.")
    plt.legend()
    plt.savefig(os.path.join(save_path, "average_feat_sz"))

    plt.clf()
    plt.cla()
    plt.close()

def MODEL_EFFICIENY(log, save_path):
    bests = log.get_prop("best_model")
    zero_v_ll = log.get_point(0, "zero_v_ll")

    efficiencies = [ s / (vl - zero_v_ll) for (s, _, _, vl,_) in bests]
    v_lls = [vl for (_,_,_,vl,_) in bests]

    pairs = [(s, vl) for (s,_,_,vl,_) in bests]
    sizes, lls = zip(*pairs)

    best_tf = np.argmax([tf for (_,_,tf,_,_) in bests])
    best_vf = np.argmax([vf for (_,vf,_,_,_) in bests])
    best_tl = np.argmax([tl for (_,_,_,_,tl) in bests])
    best_vl = np.argmax([vl for (_,_,_,vl,_) in bests])

    plt.plot(sizes, lls, "kd")
    plt.plot(sizes[best_tf], lls[best_tf], "bo", label="best TF")
    plt.plot(sizes[best_vf], lls[best_vf], "go", label="best VF")
    plt.plot(sizes[best_tl], lls[best_tl], "mo", label="best TL")
    plt.plot(sizes[best_tf], lls[best_tf], "yo", label="best TF")

    plt.legend()

    plt.savefig(os.path.join(save_path, "size_ll_plt"))

    plt.clf()
    plt.cla()
    plt.close()

    plt.subplot(2,1,1)
    plt.plot(range(1, len(efficiencies) + 1), efficiencies, "ro", label="Efficiency of best (TF)")
    plt.xlabel("Generation")
    plt.ylabel("Efficieny")
    plt.legend(loc="lower right")

    plt.subplot(2,1,2)
    plt.plot(range(1, len(v_lls) + 1), v_lls, label="LL (V) of best (TF)")
    plt.xlabel("Generation")
    plt.ylabel("LL")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_path, "efficieny_best"))

    plt.clf()
    plt.cla()
    plt.close()

def BEST_MODEL(log, save_path):

    models = log.get_prop("best_model")

    best_model_fit_t = np.argmax([fit_t for (size, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_fit_v = np.argmax([fit_v for (size, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_ll_t = np.argmax([ll_t for (size, fit_v, fit_t, ll_v, ll_t) in models])
    best_model_ll_v = np.argmax([ll_v for (size, fit_v, fit_t, ll_v, ll_t) in models])

    ind = [
        ("Fit (T)", best_model_fit_t),
        ("Fit (V)", best_model_fit_v),
        ("LL (T)", best_model_ll_t),
        ("LL (V)", best_model_ll_v)
    ]


    attrs = [(
    "According to", "Generation", "SDD Size", "Fitness (V)", "Fitness (T)", "LL (V)", "LL (T)"
    )]

    for i in ind:
        name, index  = i
        (size, fv, ft, lv, lt) = models[index]

        row = (name, index, size, fv, ft, lv, lt)

        attrs.append(row)

    print(stru.pretty_print_table(attrs))

    f = open(os.path.join(save_path, "best_models_stats"), "w")

    f.write(stru.pretty_print_table(attrs))
    f.close()

    f = open(os.path.join(save_path, "Best_models"), "w")

    for (name, index) in ind:
        f.write(f"Best model according to: {name} \n")
        #f.write(models[index][0].to_string())
        f.write("\n\n\n\n")

    f.close()



def INDICATOR_PROFILE(log, save_path):
    profile = log.get_prop("indicator_profile")
    n_gen = len(profile)

    top_points = {}
    top_x = 5

    for gen in range(n_gen):
        #1) extract:
        #   -> indicators (always the same)
        #   -> amount of times used
        indicators, amounts = profile[gen]

        #2) find the top x indices for each gen
        top_indices_of_gen = np.argsort(amounts)[-top_x:]

        #3) collect the points for each indicator
        for top_index in top_indices_of_gen:
            indicator = indicators[top_index]
            amount = amounts[top_index]

            if indicator not in top_points:
                top_points[indicator] = []

            top_points[indicator] += [(gen, amount)]


    graphs = []
    # Now to connect the points with neighbouring generations
    for (key, value) in top_points.items():
        # key: indicator
        # value: [(gen, amount)]
        new_graphs = link_generations(value)

        graphs += new_graphs

    for graph in graphs:
        x, y = zip(*graph)
        plt.plot(x, y)


    plt.xlabel("Generation")
    plt.ylabel("Amount of indicator use" )

    plt.savefig(os.path.join(save_path, "indicator_profile"))

    plt.clf()
    plt.cla()
    plt.close()

    pass

def link_generations(points):
    # Points: [(gen, amount)]

    # graphs: [[(g1, a1), ... (gn, an)], ... [...]]
    #   -> list of lists
    #   -> each list is a list of (g, a) tuples where the gn = gm + 1
    graphs = []
    current_graph = [points[0]]

    for i in range(1, len(points)):
        pr_g, _ = current_graph[-1]
        c_g, c_a = points[i]

        if c_g == pr_g + 1:
            current_graph += [(c_g, c_a)]
        else:
            graphs += [current_graph]
            current_graph = [(c_g, c_a)]

    graphs += [current_graph]

    return graphs


def LIVE_DEAD_SIZE(log, save_path):
    live = log.get_prop("live_size")
    dead = log.get_prop("dead_size")

    plt.plot(range(len(live)), live, label= "Live size")
    plt.plot(range(len(dead)), dead, label= "Dead size")

    plt.xlabel("Generation")
    plt.ylabel("Size")

    plt.legend(loc="bottom right")
    plt.savefig(os.path.join(save_path, "live_dead"))

    plt.clf()
    plt.cla()
    plt.close()

def NB_FACTORS(log, save_path):
    dat = log.get_prop("nb_factors")

    s_min = [min(gen) for gen in dat]
    s_max = [max(gen) for gen in dat]
    s_avg = [np.mean(gen) for gen in dat]
    s_best = log.get_prop("best_nb_factors")

    plt.plot(range(len(s_min)), s_min, label = "Min nb fact.")
    plt.plot(range(len(s_min)), s_max, label = "Min nb fact.")
    plt.plot(range(len(s_min)), s_avg, label = "Min nb fact.")
    plt.plot(range(len(s_min)), s_best, label= "Best model")

    plt.xlabel("Generation")
    plt.ylabel("Number")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "nb_factors"))

    plt.clf()
    plt.cla()
    plt.close()

def BEST_IND(log, save_path):
    best_fits = log.get_prop("fit_t")
    best_inds = log.get_prop("best_ind")


    max_gen = np.argmax([max(gen) for gen in best_fits])

    print(len(best_inds))
    print(len(best_fits))
    print(max_gen)
    print(best_inds[max_gen])

def BEST_AND_SIZE(log, save_path):
    size = log.get_prop("size")
    fit_t = log.get_prop("fit_t")
    fit_v = log.get_prop("fit_v")
    ll_t = log.get_prop("ll_t")
    ll_v = log.get_prop("ll_v")

    top_ind_ll_t = [np.argmax(gen) for gen in ll_t]
    top_ind_fit_v = [np.argmax(gen) for gen in fit_v]

    top_ll_t = [max(gen) for gen in ll_t]
    top_ll_v = [gen[top_ind_ll_t[i]] for (i, gen) in enumerate(ll_v)]
    top_sizes = [gen[top_ind_ll_t[i]] for (i, gen) in enumerate(size)]

    top_fit_v = [max(gen) for gen in fit_v]
    top_ll_v_ = [gen[top_ind_fit_v[i]] for (i, gen) in enumerate(ll_v)]
    top_sizes = [gen[top_ind_ll_t[i]] for (i, gen) in enumerate(size)]

    plt.plot(range(len(fit_t)), top_ll_t, label="LL(T) of Best according to T")
    plt.plot(range(len(fit_t)), top_ll_v, label="LL(V) of Best according to T")

    # TODO: hacky code
    if "base_ll_tr" in log:
        goal_ll_t = log.get_prop("base_ll_tr")[0]
        plt.plot(range(len(fit_t)), [goal_ll_t for i in range(len(fit_t))], label="Goal ll (T)")

    plt.xlabel("Generation")
    plt.ylabel("LL")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "best_ll"))

    plt.clf()
    plt.cla()
    plt.close()


    plt.subplot(2,1, 1)
    plt.plot(range(len(fit_t)), top_fit_v, label="FIT(V) of Best according to V")

    #TODO: hacky code
    if "base_fit_va" in log:
        goal_fit_v = log.get_prop("base_fit_va")[0]
        plt.plot(range(len(fit_t)), [goal_fit_v for i in range(len(fit_t))], label="Goal fit (V)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")

    plt.subplot(2,1, 2)
    plt.plot(range(len(fit_t)), top_ll_v_, label="LL(V) of Best according to V")

    #TODO: hacky code
    if "base_ll_va" in log:
        goal_ll_v = log.get_prop("base_ll_va")[0]
        plt.plot(range(len(fit_t)), [goal_ll_v for i in range(len(fit_t))], label="Goal ll (V)")

    plt.xlabel("Generation")
    plt.ylabel("LL")
    plt.legend(loc = "lower right")

    plt.savefig(os.path.join(save_path, "best_fit"))

    plt.clf()
    plt.cla()
    plt.close()

def SIZES(log, save_path):
    sizes = log.get_prop("sizes")

    size_best = log.get_prop("best_size")

    max_s = [max(it) for it in sizes]
    min_s = [min(it) for it in sizes]
    avg_s = [np.mean(it) for it in sizes]

    plt.plot(range(len(sizes)), max_s, label="Max size per iteration")
    plt.plot(range(len(sizes)), min_s, label="Min size per iteration")
    plt.plot(range(len(sizes)), avg_s, label="Average size per iteration")
    plt.plot(range(len(size_best)), size_best, label="Best model size (according to fitness)")

    # TODO: hacky code
    if "base_size" in log:
        base_s = log.get_prop("base_size")[0]
        plt.plot(range(len(sizes)), [base_s for i in range(len(sizes))], label="Goal size of SDD")

    plt.xlabel("Generation")
    plt.ylabel("Size")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "Sizes"))

    plt.clf()
    plt.cla()
    plt.close()



def TIMES(log, save_path):
    t_fit = np.mean(log.get_prop("time: fitting"))
    t_mut = np.mean(log.get_prop("time: mutation"))

    # Create time plot
    times = log.get_prop("time")
    times_fitting = log.get_prop("time: fitting")
    times_cross = log.get_prop("time: crossing")
    times_mutation = log.get_prop("time: mutation")
    times_extra = log.get_prop("time: extra")

    plt.plot(range(len(times)), times, label="Times per iteration")
    plt.plot(range(len(times)), times_fitting, label="Times per iteration (fitting)")
    plt.plot(range(len(times)), times_cross, label="Times per iteration (cross)")
    plt.plot(range(len(times)), times_mutation, label="Times per iteration (mutation)")
    plt.plot(range(len(times)), times_extra, label="Extra time")
    plt.xlabel("Generation")
    plt.ylabel("Time (s)")
    plt.legend(loc = "upper left")
    plt.savefig(os.path.join(save_path, "Times"))

    plt.clf()
    plt.cla()
    plt.close()

    print(t_fit, t_mut)

def FITNESS_VALID(log, save_path):


    max_fit_v = [max(it) for it in log.get_prop("fit_v")]
    avg_fit_v = [np.mean(it) for it in log.get_prop("fit_v")]
    min_fit_v = [min(it) for it in log.get_prop("fit_v")]

    plt.plot(range(1, len(avg_fit_v) + 1), avg_fit_v, label = "populaion fitness (V)")
    plt.plot(range(1, len(max_fit_v) + 1), max_fit_v, label = "maximum fitness (V)")
    plt.plot(range(1, len(min_fit_v) + 1), min_fit_v, label = "minimum fitness (V)")

    if "base_fit_va" in log:
        goal_fit_v = log.get_prop("base_fit_va")[0]
        plt.plot(range(1, len(max_fit_v) + 1), [goal_fit_v for i in range(len(max_fit_v))], label="Goal (V)")


    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "validation_fit_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()

def FITNESS_TRAIN(log, save_path):

    max_fit_t = [max(it) for it in log.get_prop("fit_t")]
    avg_fit_t = [np.mean(it) for it in log.get_prop("fit_t")]
    min_fit_t = [min(it) for it in log.get_prop("fit_t")]

    plt.plot(range(1, len(avg_fit_t) + 1), avg_fit_t, label = "populaion fitness (T)")
    plt.plot(range(1, len(max_fit_t) + 1), max_fit_t, label = "maximum fitness (T)")
    plt.plot(range(1, len(min_fit_t) + 1), min_fit_t, label = "minimum fitness (T)")

    if "base_fit_tr" in log:
        goal_fit_t = log.get_prop("base_fit_tr")[0]
        plt.plot(range(1, len(max_fit_t) + 1), [goal_fit_t for i in range(len(max_fit_t))], label="Goal (T)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "training_fit_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()

def LL_TRAIN(log, save_path):


    min_ll_t = [min(it) for it in log.get_prop("ll_t")]
    max_ll_t = [max(it) for it in log.get_prop("ll_t")]
    avg_ll_t = [np.mean(it) for it in log.get_prop("ll_t")]

    plt.plot(range(1, len(min_ll_t) + 1), min_ll_t, label="min LL (T)")
    plt.plot(range(1, len(avg_ll_t) + 1), avg_ll_t, label="pop LL (T)")
    plt.plot(range(1, len(max_ll_t) + 1), max_ll_t, label="max LL (T)")

    if "base_ll_tr" in log:
        goal_ll_t = log.get_prop("base_ll_tr")[0]
        plt.plot(range(1, len(max_ll_t) + 1), [goal_ll_t for i in range(len(max_ll_t))], label="Goal (T)")

    plt.xlabel("Generation")
    plt.ylabel("Log Liklihood")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, "training_ll_plot.pdf"))

    plt.clf()
    plt.cla()
    plt.close()


def LL_VALID(log, save_path):

    min_ll_v = [min(it) for it in log.get_prop("ll_v")]
    max_ll_v = [max(it) for it in log.get_prop("ll_v")]
    avg_ll_v = [np.mean(it) for it in log.get_prop("ll_v")]

    plt.plot(range(1, len(min_ll_v) + 1), min_ll_v, label="min LL (V)")
    plt.plot(range(1, len(avg_ll_v) + 1), avg_ll_v, label="pop LL (V)")
    plt.plot(range(1, len(max_ll_v) + 1), max_ll_v, label="max LL (V)")

    if "base_ll_va" in log:
        goal_ll_v = log.get_prop("base_ll_va")[0]
        plt.plot(range(1, len(max_ll_v) + 1), [goal_ll_v for i in range(len(max_ll_v))], label="Goal (V)")

    plt.xlabel("Generation")
    plt.ylabel("Log Liklihood")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(save_path, 'validation_ll_plot.pdf'))

    plt.clf()
    plt.cla()
    plt.close()
