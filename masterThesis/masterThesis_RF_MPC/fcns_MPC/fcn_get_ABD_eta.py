import numpy as np
from fcns.hatMap import hatMap
from fcns.vec import vec

def fcn_get_ABD_eta(Xt, Ut, p):
     # linear dynamics for rotation
     # evolution variable is eta

     ## parameters
     dt = p['Tmpc']

     ## unpack
     xop = Xt[0:3]
     vop = Xt[3:6]
     Rop = np.reshape(Xt[6:15], (3, 3), order='F')
     wop = Xt[15:18]
     pf34 = np.reshape(Xt[18:30], (3, 4), order='F')

     ## constants for linear matrices
     # [x,v,eta,w,constant]
     [Cx_x, Cx_v, Cv_v, Cv_u, Cv_c] = eta_co_xv(Ut, dt, p['mass'], p['g'])
     [CE_eta, CE_w, CE_c] = eta_co_R(Rop, wop, dt)
     [Cw_x, Cw_eta, Cw_w, Cw_u, Cw_c] = eta_co_w(xop, Rop, wop, Ut, dt, p['J'], pf34)

     ## Assemble matrices
     # A - (12, 12)
     # B - (12, 12)
     # D - (12, )
     # A * X + B * U = D
     A = np.block([[Cx_x, Cx_v, np.zeros((3, 6))],
                    [np.zeros((3, 3)), Cv_v, np.zeros((3, 6))],
                    [np.zeros((3, 6)), CE_eta, CE_w],
                    [Cw_x, np.zeros((3, 3)), Cw_eta, Cw_w]])
     B = np.block([[np.zeros((3, 12))],
                    [Cv_u],
                    [np.zeros((3, 12))],
                    [Cw_u]])
     D = np.block([[np.zeros((3, 1))],
                    [Cv_c],
                    [CE_c],
                    [Cw_c]])
     D = np.squeeze(D)

     return A, B, D

## Aux fcns
def fcn_get_F(k):
     F = np.block([[k, np.zeros((6, ))], 
                    [np.zeros((3, )), k, np.zeros((3, ))],
                    [np.zeros((6, )), k]])
     return F

def fcn_get_N():
     N = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0], 
                         [0, 0, -1], 
                         [0, 0, 0], 
                         [1, 0, 0], 
                         [0, 1, 0], 
                         [-1, 0, 0], 
                         [0, 0, 0]])
     return N

def fcn_get_D(in_vec):
     # equal to [- N @ hatMap(in_vec)]
     d, e, f = in_vec
     D = np.array([[0, 0, 0], 
                         [e, -d, 0], 
                         [f, 0, -d], 
                         [-e, d, 0], 
                         [0, 0, 0], 
                         [0, f, -e], 
                         [-f, 0, d], 
                         [0, -f, e], 
                         [0, 0, 0]])
     return D

## Core fcns for constant matrix
def eta_co_xv(fop, dt, mass, g):
     Cx_x = np.eye(3)
     Cx_v = np.eye(3) * dt

     Cv_v = np.eye(3)
     Cv_u = dt / mass * np.hstack([np.eye(3) for _ in range(4)]) # (3 ,12)
     Cv_c = Cv_u @ fop + np.array([0, 0, -g]) * dt # (3, )
     Cv_c = np.reshape(Cv_c, (3, 1), order='F') # (3, 1)

     return Cx_x, Cx_v, Cv_v, Cv_u, Cv_c

def eta_co_R(Rop, wop, dt):
    # the input arguments are composed of variables at the operating point
    # and parameters

    N = fcn_get_N()

    ## debugged code
    invN = np.linalg.pinv(N)

    C_eta = np.kron(np.eye(3), Rop @ hatMap(wop)) @ N + np.kron(np.eye(3), Rop) @ fcn_get_D(wop) # (9, 3)
    C_w = np.kron(np.eye(3), Rop) @ N # (9, 3)
    C_c = vec(Rop @ hatMap(wop)) - np.kron(np.eye(3), Rop) @ N @ wop # (9, )

    CE_eta = np.eye(3) + invN * dt @ np.kron(np.eye(3), Rop.T) @ C_eta # (3, 3)
    CE_w = invN * dt @ np.kron(np.eye(3), Rop.T) @ C_w # (3, 3)
    CE_c = invN * dt @ np.kron(np.eye(3), Rop.T) @ C_c # (3, )
    CE_c = np.reshape(CE_c, (3, 1), order='F') # (3, 1)

    return CE_eta, CE_w, CE_c

def eta_co_w(xop, Rop, wop, fop, dt, J, pf):
    # the input arguments are composed of variables at the operating point 
    # and parameters

    N = fcn_get_N()
    r1 = pf[:,0] - xop
    r2 = pf[:,1] - xop
    r3 = pf[:,2] - xop
    r4 = pf[:,3] - xop
    Mop = np.hstack((hatMap(r1), hatMap(r2), hatMap(r3), hatMap(r4))) @ fop # (3, 1)

    temp_J_w = hatMap(J @ wop) - hatMap(wop) @ J # (3, 3)
    sum_fop = np.hstack([np.eye(3) for _ in range(4)]) @ fop # (3, )

    Cx = Rop.T @ hatMap(sum_fop) # (3, 3)
    Ceta = fcn_get_F(Mop) @ N - temp_J_w @ hatMap(wop)  # (3, 3)
    Cw = temp_J_w  # (3, 3)
    Cu = Rop.T @ np.hstack((hatMap(r1), hatMap(r2), hatMap(r3), hatMap(r4)))  # (3, 12)
    Cc = -hatMap(wop) @ J @ wop + Rop.T @ Mop - temp_J_w @ wop - Cx @ xop  # (3, )

    Cw_x = dt * (np.linalg.inv(J) @ Cx) # (3, 3)
    Cw_eta = dt * (np.linalg.inv(J) @ Ceta)  # (3, 3)
    Cw_w = dt * (np.linalg.inv(J) @ Cw) + np.eye(3)  # (3, 3)
    Cw_u = dt * (np.linalg.inv(J) @ Cu)  # (3, 12)
    Cw_c = dt * (np.linalg.inv(J) @ Cc)  # (3, )
    Cw_c = np.reshape(Cw_c, (3, 1), order='F') # (3, 1)

    return Cw_x, Cw_eta, Cw_w, Cw_u, Cw_c