import numpy as np

def vec(m):
    v = np.reshape(m, (np.size(m),), order='F')

    return v