import argparse

import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def append_dict(dictionary, new_dict):
    for key in new_dict:
        if key not in dictionary:
            dictionary[key]=[]
        dictionary[key].append(new_dict[key])

def mean_dict(dictionary):
    out_dict = dict()
    for key in dictionary:
        if isinstance(dictionary[key], list):
            if isinstance(dictionary[key][0], list):
                out_dict[key] = [np.mean([dictionary[key][j][i] for j in range(len(dictionary[key]))]) for i in range(len(dictionary[key][0]))]
            else:
                out_dict[key] = np.mean(np.array(dictionary[key]))
        else:
            out_dict[key] = dictionary[key]
    return out_dict
