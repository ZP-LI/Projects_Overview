import numpy as np

def bz_int(alpha, x0=0, s_max=1):

    # make sure "alpha" is a row vector (row > column)
    try:
        alpha.shape[1]
    except:
        None
    else:
        if alpha.shape[0] > alpha.shape[1]:
            alpha = alpha.T
        alpha = np.squeeze(alpha)

    M = len(alpha)
    AA = np.zeros((M+1,M+1))

    for ii in range(M):
        AA[ii,ii:ii+2] = [-1, 1]

    AA = M / s_max * AA
    AA[M,0] = 1

    b = np.zeros((M+1,))
    b[0:-1] = alpha
    b[-1] = x0

    alpha_int = np.around(np.linalg.solve(AA, b), decimals=4)

    return alpha_int