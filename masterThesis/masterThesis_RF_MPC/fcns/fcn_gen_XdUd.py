import numpy as np
from scipy.linalg import expm
from fcns.hatMap import hatMap

def fcn_gen_XdUd(t, Xt, bool_inStance, p):
    # parameters
    gait = p['gait']
    acc_d = p['acc_d']
    vel_d = p['vel_d']
    yaw_d = p['yaw_d']
    
    # generate reference trajectory
    # X = [pc dpc eta wb]
    lent = len(t) # ???
    if lent == 1:
        Xd = np.zeros((30,))
        Ud = np.zeros((12,))
    else:
        Xd = np.zeros((30,lent))
        Ud = np.zeros((12,lent))

    Rground = p['Rground']           # ground slope

    for ii in range(lent):
        if gait >= 0:        # March forward and rotate
            # linear motion
            pc_d = np.array([0.0, 0.0, p['z0']])
            dpc_d = np.array([0.0, 0.0, 0.0])
            for jj in range(2):
                if t[ii] < (vel_d[jj] / acc_d):
                    dpc_d[jj] = acc_d * t[ii]
                    pc_d[jj] = 1/2 * acc_d * t[ii]**2
                else:
                    dpc_d[jj] = vel_d[jj]
                    pc_d[jj] = vel_d[jj] * t[ii] - 1/2 * vel_d[jj] * vel_d[jj]/acc_d

            # angular motion
            if Xt is None:
                ea_d = np.array([0.0, 0.0, 0.0])
            else:
                ea_d = np.array([0.0, 0.0, yaw_d])

            vR_d = expm(hatMap(ea_d)).reshape((9,), order='F')
            wb_d = np.array([0, 0, 0])

        pfd = (Rground @ p['pf34']).reshape((12,), order='F')

        if lent == 1:
            Xd = np.concatenate((pc_d, dpc_d, vR_d, wb_d, pfd))
        else:
            Xd[:, ii] = np.concatenate((pc_d, dpc_d, vR_d, wb_d, pfd))
        
        # force
        # if gait == -3:
        #     Ud[:, ii] = U_d
        # else:
        sum_inStance = np.sum(bool_inStance[:, ii])
        if sum_inStance == 0:    # four legs in swing
            if lent == 1:
                Ud = np.zeros(12)
            else:
                Ud[:, ii] = np.zeros(12)
        else:
            if lent == 1:
                Ud[[2, 5, 8, 11]] = bool_inStance[:, ii] * (p['mass'] * p['g'] / sum_inStance)
            else:
                Ud[[2, 5, 8, 11], ii] = bool_inStance[:, ii] * (p['mass'] * p['g'] / sum_inStance)

    return Xd, Ud












