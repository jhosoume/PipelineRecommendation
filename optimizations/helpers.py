import pickle

def save_opt(opt, name):
    with open("optimizations/opt_{}.pickle".format(name), 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_opt(name):
    with open("optimizations/opt_{}.pickle".format(name), 'rb') as handle:
        opt = pickle.load(handle)
    return opt
