import numpy as np

def hatMap(x):
    x = np.squeeze(x)

    if len(x) != 3:
        raise ValueError("Input should be a 3D vector !")

    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
