import numpy as np
from scipy.linalg import expm

from fcns.polyval_bz import polyval_bz
from fcns.hatMap import hatMap
from fcns.fcn_gen_XdUd import fcn_gen_XdUd
# from fcns_MPC.fcn_bound_ref_traj import fcn_bound_ref_traj

def fcn_FSM(t_, Xt, p):

    # parameters
    L, W, d = p['L'], p['W'], p['d']
    gait = p['gait']
    Tst_ = p['Tst']

    if np.absolute(np.linalg.norm(Xt[3:5])) < 0.0001:
        Tst = Tst_
    else:
        Tst = min(Tst_,0.2/np.linalg.norm(Xt[3:5]))

    Tsw = p['Tsw']
    T = Tst + Tsw
    Tair = 1/2 * (Tsw - Tst)

    pc, dpc, vR, wb, pf = np.split(Xt, [3, 6, 15, 18])
    R = np.reshape(vR, [3, 3], order='F')
    idx_pf = np.arange(18, 30)
    pf34 = np.reshape(pf, [3, 4], order='F')

    # initialization
    global FSM, Ta, Tb, pf_R_trans
    try:
        FSM
    except:
        FSM = np.zeros((4,))
        Ta = np.zeros((4,))
        Tb = np.ones((4,))
        pf_R_trans = Xt[idx_pf]

    t = t_[0] # current time
    s = np.zeros((4,))


    ## FSM
    # 1 - stance
    # 2 - swing

    for i_leg in range(4):
        s[i_leg] = (t - Ta[i_leg]) / (Tb[i_leg] - Ta[i_leg])
        s[s < 0] = 0
        s[s > 1] = 1

        # --- FSM ---
        if FSM[i_leg] == 0:          # init to stance
            if gait == -1:           # pose control
                Ta[i_leg] = 0
                Tb[i_leg] = -1
            elif gait == 1:        # bound
                Ta[0:2] = [t,t]
                Ta[2:4] = np.array([1,1]) * (t + 1/2*(Tst + Tsw))
                Tb[0:2] = Ta[0:2] + Tst
                Tb[2:4] = Ta[2:4] + Tst + Tair
            elif gait == 2:        # pacing
                Ta[[0,2]] = [t,t]
                Ta[[1,3]] = np.array([1,1]) * (t + 1/2*(Tst + Tsw))
                Tb[i_leg] = Ta[i_leg] + Tst
            elif gait == 3:        # gallop
                Ta[0] = t
                Ta[1] = t + 0.05
                Ta[2] = t + 0.05 + Tst
                Ta[3] = t + 0.1 + Tst
                Tb[i_leg] = Ta[i_leg] + Tst
            elif gait == 5:        # crawl
                Ta[0] = t
                Ta[1] = t + Tsw
                Ta[2] = t + Tsw*2
                Ta[3] = t + Tsw*3
                Tb[i_leg] = Ta[i_leg] + Tst
            else:                    # trot walk
                Ta[[0,3]] = [t,t]
                Ta[[1,2]] = np.array([1,1]) * (t + 1/2*(Tst + Tsw))
                Tb[i_leg] = Ta[i_leg] + Tst

            FSM[i_leg] = FSM[i_leg] + 1
            pf_R_trans = Xt[idx_pf]

        elif FSM[i_leg] == 1 and (s[i_leg] >= 1 - 1e-7):  # stance to swing
            FSM[i_leg] = FSM[i_leg] + 1
            Ta[i_leg] = t
            Tb[i_leg] = Ta[i_leg] + Tsw
            pf_R_trans = Xt[idx_pf]

        elif FSM[i_leg] == 2 and (s[i_leg] >= 1 - 1e-7):  # swing to stance
            FSM[i_leg] = 1
            Ta[i_leg] = t
            Tb[i_leg] = Ta[i_leg] + Tst
            pf_R_trans = Xt[idx_pf]

    s = (t - Ta) / (Tb - Ta)
    s[s < 0] = 0
    s[s > 1] = 1

    # FSM in prediction horizon
    FSM_ = np.tile(FSM, (p['predHorizon'], 1)).T
    for i_leg in range(4):
        for ii in range(1, p['predHorizon']):
            if t_[ii] <= Ta[i_leg]:
                FSM_[i_leg, ii] = 1
            elif Ta[i_leg] < t_[ii] < Tb[i_leg]:
                FSM_[i_leg, ii] = FSM[i_leg]
            elif Ta[i_leg] + Tst + Tsw < t_[ii]:
                FSM_[i_leg, ii] = FSM[i_leg]
            else:
                if FSM[i_leg] == 1:
                    FSM_[i_leg, ii] = 2
                else:
                    FSM_[i_leg, ii] = 1

    if gait == -1:      # pose
        FSM_ = np.ones_like(FSM_)

    # [4, predHorizon]: bool matrix
    bool_inStance = (FSM_ == 1)

    ## Gen ref traj
    # --- Xd/Ud ---
    [Xd,Ud] = fcn_gen_XdUd(t_,Xt,bool_inStance,p)

    # p = fcn_bound_ref_traj(p)

    if gait == 1:   # bound
        for ii in range(p['predHorizon']):
            fsm = FSM_[:, ii]
            if fsm[0] == 1:     # front stance
                s_ph = (t_[ii] - Ta[0]) / (Tb[0] - Ta[0])

                th_d = polyval_bz(-p['th_co'], s_ph)
                dth_d = polyval_bz(-p['dth_co'], s_ph)
                z_d = polyval_bz(p['z_co'], s_ph)
                vR_d = np.reshape(expm(hatMap([0, th_d, 0])), [9, 1], order='F')
                Xd[2, ii] = z_d
                Xd[6:15, ii] = vR_d
                Xd[16, ii] = dth_d

                Fz_d = polyval_bz(p['Fz_co'], s_ph)
                tau_d = polyval_bz(p['tau_co'], s_ph)
                r = pf34[:, 0] - pc
                Ud[[2, 5], ii] = 0.5*Fz_d
                Ud[[0, 3], ii] = 0.5*(r[0]*Fz_d - tau_d) / r[2]

            elif fsm[2] == 1:   # back stance
                s_ph = (t_[ii] - Ta[2]) / (Tb[2] - Ta[2])

                th_d = polyval_bz(p['th_co'], s_ph)
                dth_d = polyval_bz(p['th_co'], s_ph)
                z_d = polyval_bz(p['z_co'], s_ph)
                vR_d = np.reshape(expm(hatMap([0, th_d, 0])), [9, 1], order='F')
                Xd[2, ii] = z_d
                Xd[6:15, ii] = vR_d
                Xd[16, ii] = dth_d

                Fz_d = polyval_bz(p['Fz_co'], s_ph)
                tau_d = polyval_bz(-p['tau_co'], s_ph)
                r = pf34[:, 2] - pc
                Ud[[8, 11], ii] = 0.5*Fz_d
                Ud[[6, 9], ii] = 0.5*(r[0]*Fz_d - tau_d) / r[2]

    ## swing leg
    p_hip_b = np.array([[L/2, W/2+d, 0], [L/2, -W/2-d, 0], [-L/2, W/2+d, 0], [-L/2, -W/2-d, 0]]).T
    p_hip_R = R @ p_hip_b
    ws = R @ wb
    v_hip_R = np.tile(dpc, (4, 1)).T + hatMap(ws) @ p_hip_R

    # capture point
    # p_cap = Tsw/2 * vd + sqrt(z0/g) * (vc - vd)
    p_cap = np.zeros((2,4))
    vd = Xd[3:5, 0]
    for i_leg in range(4):
        temp = 0.8 * Tst * vd + np.sqrt(p['z0'] / p['g']) * (v_hip_R[0:2, i_leg] - vd)
        temp[temp < -0.15] = -0.15
        temp[temp > 0.15] = 0.15
        p_cap[:, i_leg] = pc[0:2] + p_hip_R[0:2, i_leg] + temp

    # desired foot placement
    pfd = Xt[idx_pf]
    for i_leg in range(4):
        idx = 3 * i_leg + np.arange(1, 4) - 1
        if FSM[i_leg] == 2:
            co_x = np.linspace(pf_R_trans[idx[0]], p_cap[0, i_leg], 6)
            co_y = np.linspace(pf_R_trans[idx[1]], p_cap[1, i_leg], 6)
            co_z = np.array([0, 0, 0.1, 0.1, 0, -0.002])
            pfd[idx] = np.array([polyval_bz(co_x, s[i_leg]),
                                polyval_bz(co_y, s[i_leg]),
                                polyval_bz(co_z, s[i_leg])])

    Xd[idx_pf, :] = np.tile(pfd, (p['predHorizon'], 1)).T

    # output
    FSMout = FSM

    return FSMout, Xd, Ud, Xt
















