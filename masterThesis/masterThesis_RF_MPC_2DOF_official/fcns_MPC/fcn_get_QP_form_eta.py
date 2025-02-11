import numpy as np
from scipy.linalg import logm, expm
from fcns_MPC.fcn_get_ABD_eta import fcn_get_ABD_eta
from fcns.veeMap import veeMap

def fcn_get_QP_form_eta(Xt, Ut, Xd, Ud, p):
    # min. 0.5 * x' * H *x + g' * x
    # s.t. Aineq *x <= bineq
    #      Aeq * x <= beq
    # X = [pc dpc vR wb pf]': [30,1]
    # q = [pc dpc eta wb]: [12 1]
    # lb/ub - [4,n_hor]

    ## parameters
    mu = p['mu']
    n_hor = p['predHorizon']
    Umax = p['Umax']
    decayRate = p['decayRate']

    R = p['R']
    Q = p['Q']
    Qf = p['Qf']
    Qx, Qv, Qeta, Qw = Q[0:3, 0:3], Q[3:6, 3:6], Q[6:9, 6:9], Q[9:12, 9:12]
    Qxf, Qvf, Qetaf, Qwf = Qf[0:3, 0:3], Qf[3:6, 3:6], Qf[6:9, 6:9], Qf[9:12, 9:12]

    nX = 12
    nU = 12

    ## A,B,d matrices for linear dynamics
    [A, B, d] = fcn_get_ABD_eta(Xt, Ut, p)

    ## Decompose
    Rt = np.reshape(Xt[6:15], (3, 3), order='F')
    qt = np.concatenate((Xt[0:6], [0, 0, 0], Xt[15:18])) # 3rd-Item (rotation matric [3,3] -> eta [3, ])

    # lb <= Fz <= ub
    Fzd = Ud[[2, 5, 8, 11], :]
    lb = -1 * Fzd
    ub = 2 * Fzd

    ## Matrices for QP
    H = np.zeros(((nX + nU) * n_hor, (nX + nU) * n_hor)) # (24*7, 24*7)
    g = np.zeros((H.shape[0], )) # (24*7, )
    Aeq = np.zeros((nX * n_hor, (nX + nU) * n_hor)) # (12*7, 24*7)
    beq = np.zeros((Aeq.shape[0], )) # (12*7, )

    if p['gait'] == -2:
        Aineq_unit = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    else:
        Aineq_unit = np.array([[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu], [0, 0, 1], [0, 0, -1]])

    nAineq_unit = Aineq_unit.shape[0]
    Aineq = np.zeros((4 * nAineq_unit * n_hor, (nX + nU) * n_hor)) # (4*6*7, 24*7)
    bineq = np.zeros((Aineq.shape[0], )) # (4*6*7, )

    for i_hor in range(n_hor):
        xd = Xd[0:3, i_hor]
        vd = Xd[3:6, i_hor]
        Rd = np.reshape(Xd[6:15, i_hor], (3, 3), order='F')
        wd = Xd[15:18, i_hor]

        ## Objective function
        idx_u = i_hor * (nX + nU) + np.arange(0, nU)
        idx_x = i_hor * (nX + nU) + nU + np.arange(0, nX)
        if i_hor == n_hor - 1:
            H[idx_x[0]:idx_x[-1]+1, idx_x[0]:idx_x[-1]+1] = Qf * decayRate**i_hor
            g[idx_x] = np.concatenate((-Qxf @ xd,
                                        -Qvf @ vd,
                                        Qetaf @ veeMap(logm(Rd.T @ Rt)),
                                        -Qwf @ wd)) * decayRate**i_hor
        else:
            H[idx_x[0]:idx_x[-1]+1, idx_x[0]:idx_x[-1]+1] = Q * decayRate**i_hor
            g[idx_x] = np.concatenate((-Qx @ xd,
                                        -Qv @ vd,
                                        Qeta @ veeMap(logm(Rd.T @ Rt)),
                                        -Qw @ wd)) * decayRate**i_hor
        H[idx_u[0]:idx_u[-1]+1, idx_u[0]:idx_u[-1]+1] = R * decayRate**i_hor
        g[idx_u] = R.T @ (Ut - Ud[:, i_hor]) * decayRate**i_hor

        ## Equality constraints
        if i_hor == 0:
            Aeq[0:nX, 0:(nU+nX)] = np.concatenate((-B, np.eye(nX)), axis=1)
            beq[0:nX] = A @ qt + d
        else:
            Aeq[i_hor*nX : i_hor*nX+nX, (i_hor-1)*(nX+nU)+nU : (i_hor-1)*(nX+nU)+nU+2*nX+nU] = np.concatenate((-A, -B, np.eye(nX)), axis=1)
            beq[i_hor*nX+np.arange(0, nX)] = d

        ## Inequality constraints
        Fi = np.zeros((4*nAineq_unit, 12)) # (4*6, 12)
        hi = np.zeros((Fi.shape[0], )) # (4*6, )
        for i_leg in range(4):
            idx_F = i_leg * nAineq_unit + np.arange(0, nAineq_unit)
            idx_u = i_leg * 3 + np.arange(0, 3)
            Fi[idx_F[0]:idx_F[-1]+1, idx_u[0]:idx_u[-1]+1] = Aineq_unit

            if p['gait'] == -2:
                hi[idx_F] = np.array([Umax-Ut[idx_u[0]], Umax+Ut[idx_u[0]],
                                    Umax-Ut[idx_u[1]], Umax+Ut[idx_u[1]],
                                    Umax-Ut[idx_u[2]], Umax+Ut[idx_u[2]]])
            else:
                hi[idx_F] = np.array([mu*Ut[idx_u[2]]-Ut[idx_u[0]], 
                                    mu*Ut[idx_u[2]]+Ut[idx_u[0]],
                                    mu*Ut[idx_u[2]]-Ut[idx_u[1]], 
                                    mu*Ut[idx_u[2]]+Ut[idx_u[1]],
                                    ub[i_leg, i_hor]-Ut[idx_u[2]]+Ud[idx_u[2], i_hor],
                                    -lb[i_leg, i_hor]+Ut[idx_u[2]]-Ud[idx_u[2], i_hor]])

        idx_A = i_hor * 4*nAineq_unit + np.arange(0, 4*nAineq_unit)
        idx_z = i_hor * (nX + nU) + np.arange(0, nU)
        Aineq[idx_A[0]:idx_A[-1]+1, idx_z[0]:idx_z[-1]+1] = Fi
        bineq[idx_A] = hi

    return H, g, Aineq, bineq, Aeq, beq