import numpy as np

def veeMap(in_vec):
    # contrary function as the function "hatMap"

    out = np.zeros((3, ))

    out[0] = - in_vec[1, 2]
    out[1] = in_vec[0, 2]
    out[2] = - in_vec[0, 1]

    return out