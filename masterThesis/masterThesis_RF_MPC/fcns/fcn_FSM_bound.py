import numpy as np
from scipy.linalg import expm

# from fcns_MPC.fcn_bound_ref_traj import fcn_bound_ref_traj
from fcns.polyval_bz import polyval_bz
from fcns.hatMap import hatMap

def fcn_FSM_bound(t_, Xt, p):
    ## parameters
    acc = p['acc_d']
    vd = p['vel_d']

    Tst_ = p['Tst']

    if np.absolute(np.linalg.norm(Xt[3:5])) < 0.0001:
        Tst = Tst_
    else:
        Tst = min(Tst_,0.2/np.linalg.norm(Xt[3:5]))

    Tsw = p['Tsw']
    T = Tst + Tsw
    Tair = 1/2 * (Tsw - Tst)

    # periodic traj for bounding
    # p = fcn_bound_ref_traj(p)

    # decompose state
    pc, dpc, vR, wb, pf = np.split(Xt, [3, 6, 15, 18])
    R = np.reshape(vR, [3, 3], order='F')
    idx_pf = np.arange(18, 30)
    pf34 = np.reshape(pf, [3, 4], order='F')

    # initialization
    global FSM, Ta, Tb, pf_trans, Ta_sw, Tb_sw
    try:
        FSM
    except:
        FSM = 1
        Ta = 0
        Tb = Ta + Tst
        pf_trans = pf
        Ta_sw = np.zeros(4,)
        Tb_sw = np.zeros(4,)
        Ta_sw[0:2] = Tst - T
        Tb_sw[0:2] = Ta_sw[0:2] + Tsw
        Ta_sw[2:4] = -Tair
        Tb_sw[2:4] = Ta_sw[2:4] + Tsw

    ## FSM
    # 1 - FrontStance -- Tst expires --> 2
    # 2 - air_1 -- Tair expires --> 3
    # 3 - BackStance -- Tst expires --> 4
    # 4 - air_2 -- Tair expires --> 1

    t = t_[0]
    s = (t - Ta) / (Tb - Ta)
    if (FSM == 1) and (s >= 1):
        Ta = Tb
        Tb = Tb + Tair
        FSM = FSM + 1
        Ta_sw[0:2] = Ta_sw[0:2] + T
        Tb_sw[0:2] = Ta_sw[0:2] + Tsw
        pf_trans[0:6] = pf[0:6]
    elif (FSM == 2) and (s >= 1):
        Ta = Tb
        Tb = Tb + Tst
        FSM = FSM + 1
    elif (FSM == 3) and (s >= 1):
        Ta = Tb
        Tb = Tb + Tair
        FSM = FSM + 1
        Ta_sw[2:4] = Ta_sw[2:4] + T
        Tb_sw[2:4] = Ta_sw[2:4] + Tsw
        pf_trans[6:12] = pf[6:12]
    elif (FSM == 4) and (s >= 1):
        Ta = Tb
        Tb = Tb + Tst
        FSM = 1

    ## Xd/Ud
    s = (t - Ta) / (Tb - Ta)
    s = max(0, min(1, s))

    FSM_ = np.zeros(p['predHorizon'])
    s_ = np.zeros(p['predHorizon'])

    Ud = np.zeros((12, p['predHorizon']))
    Xd = np.tile(Xt, (p['predHorizon'], 1)).T

    pd = np.zeros((2,))
    dpd = np.zeros((2,))
    
    # reference force / position trajectory
    # Xd/Ud along the prediction horizon
    for ii in range(p['predHorizon']):
        FSM_[ii], s_[ii] = fcn_FSM_pred_hor(FSM, Ta, t_[ii], p)
        FSM_[0] = FSM
        s_[0] = s

        for dir_xy in range(2):
            if t_[ii] < (vd[dir_xy] / acc):
                dpd[dir_xy] = acc * t_[ii]
                pd[dir_xy] = 1/2 * acc * t_[ii]**2
            else:
                dpd[dir_xy] = vd[dir_xy]
                pd[dir_xy] = vd[dir_xy] * t_[ii] - 1/2 * vd[dir_xy]**2 / acc

        if FSM_[ii] == 1:  # 1 - front stance
            Fz_d = polyval_bz(p['Fz_co'], s_[ii])
            dz_d = polyval_bz(p['dz_co'], s_[ii])
            z_d = polyval_bz(p['z_co'], s_[ii])

            tau_d = polyval_bz(p['tau_co'], s_[ii])
            dth_d = polyval_bz(p['dth_co'], s_[ii])
            th_d = polyval_bz(p['th_co'], s_[ii])
            R_d = expm(hatMap([0, th_d, 0]))

            dx = pc[0] - pf34[0, 0]
            Fx_d = (dx * Fz_d - tau_d) / z_d

            Ud[0:6, ii] = 1/2 * np.concatenate(([Fx_d], [0], [Fz_d], [Fx_d], [0], [Fz_d]))
            Xd[0:18, ii] = np.concatenate([pd, [z_d], dpd, [dz_d], R_d.flatten('F'), [0], [dth_d], [0]])

        elif FSM_[ii] == 2:  # 2 - air 1
            dz_d = p['dz_co'][-1] - p['g'] * (s_[ii] * Tair)
            z_d = p['z_co'][-1] + p['dz_co'][-1] * (s_[ii] * Tair) - 1/2 * p['g'] * (s_[ii] * Tair)**2

            dth_d = p['dth_co'][-1]
            th_d = p['th_co'][-1] + dth_d * (s_[ii] * Tair)
            R_d = expm(hatMap([0, th_d, 0]))

            Xd[0:18, ii] = np.concatenate([pd, [z_d], dpd, [dz_d], R_d.flatten('F'), [0], [dth_d], [0]])

        elif FSM_[ii] == 3:  # 3 - back stance
            Fz_d = polyval_bz(p['Fz_co'], s_[ii])
            dz_d = polyval_bz(p['dz_co'], s_[ii])
            z_d = polyval_bz(p['z_co'], s_[ii])

            tau_d = polyval_bz(-p['tau_co'], s_[ii])
            dth_d = polyval_bz(-p['dth_co'], s_[ii])
            th_d = polyval_bz(-p['th_co'], s_[ii])
            R_d = expm(hatMap([0, th_d, 0]))

            dx = pc[0] - pf34[0, 2]
            Fx_d = (dx * Fz_d - tau_d) / z_d

            Ud[6:12, ii] = 1/2 * np.concatenate(([Fx_d], [0], [Fz_d], [Fx_d], [0], [Fz_d]))
            Xd[0:18, ii] = np.concatenate([pd, [z_d], dpd, [dz_d], R_d.flatten('F'), [0], [dth_d], [0]])
        
        elif FSM_[ii] == 4:  # 4 - air 2
            dz_d = p['dz_co'][-1] - p['g'] * (s_[ii] * Tair)
            z_d = p['z_co'][-1] + p['dz_co'][-1] * (s_[ii] * Tair) - 1/2 * p['g'] * (s_[ii] * Tair)**2

            dth_d = - p['dth_co'][-1]
            th_d = - p['th_co'][-1] + dth_d * (s_[ii] * Tair)
            R_d = expm(hatMap([0, th_d, 0]))

            Xd[0:18, ii] = np.concatenate([pd, [z_d], dpd, [dz_d], R_d.flatten('F'), [0], [dth_d], [0]])

    ## reference swing leg trajectory
    L, W, d = p['L'], p['W'], p['d']
    p_hip_b = np.array([[L/2, W/2+d, 0], [L/2, -W/2-d, 0], [-L/2, W/2+d, 0], [-L/2, -W/2-d, 0]]).T
    p_hip_R = R @ p_hip_b
    ws = R @ wb
    v_hip_R = np.tile(dpc, (4, 1)).T + hatMap(ws) @ p_hip_R

    # capture point
    # p_cap = Tsw/2 * vd + sqrt(z0/g) * (vc - vd)
    p_cap = np.zeros((2, 4))
    dpd = Xd[3:5, 0]
    for i_leg in range(4):
        temp = 0.8 * Tst * dpd + np.sqrt(p['z0'] / p['g']) * (v_hip_R[0:2, i_leg] - dpd)
        temp[temp < -0.15] = -0.15
        temp[temp > 0.15] = 0.15
        p_cap[:, i_leg] = pc[0:2] + p_hip_R[0:2, i_leg] + temp # 2 (x, y) * 4 (four legs)

    # desired foot placement
    pfd = Xt[idx_pf]
    s_sw = (t - Ta_sw) / (Tb_sw - Ta_sw)
    s_sw[s_sw < 0] = 0
    s_sw[s_sw > 1] = 1

    if FSM == 1:
        for i_leg in np.arange(2, 4):
            idx = 3 * i_leg + np.arange(1, 4) - 1 # (6,7,8) || (9,10,11)
            co_x = np.linspace(pf_trans[idx[0]], p_cap[0, i_leg], 6)
            co_y = np.linspace(pf_trans[idx[1]], p_cap[1, i_leg], 6)
            co_z = np.array([0, 0, 0.15, 0.15, 0, -0.002])
            pfd[idx] = np.array([polyval_bz(co_x, s_sw[i_leg]),
                                polyval_bz(co_y, s_sw[i_leg]),
                                polyval_bz(co_z, s_sw[i_leg])])
    elif FSM == 2:
        for i_leg in range(4):
            idx = 3 * i_leg + np.arange(1, 4) - 1
            co_x = np.linspace(pf_trans[idx[0]], p_cap[0, i_leg], 6)
            co_y = np.linspace(pf_trans[idx[1]], p_cap[1, i_leg], 6)
            co_z = np.array([0, 0, 0.15, 0.15, 0, -0.002])
            pfd[idx] = np.array([polyval_bz(co_x, s_sw[i_leg]),
                                polyval_bz(co_y, s_sw[i_leg]),
                                polyval_bz(co_z, s_sw[i_leg])])
    elif FSM == 3:
        for i_leg in range(2):
            idx = 3 * i_leg + np.arange(1, 4) - 1
            co_x = np.linspace(pf_trans[idx[0]], p_cap[0, i_leg], 6)
            co_y = np.linspace(pf_trans[idx[1]], p_cap[1, i_leg], 6)
            co_z = np.array([0, 0, 0.15, 0.15, 0, -0.002])
            pfd[idx] = np.array([polyval_bz(co_x, s_sw[i_leg]),
                                polyval_bz(co_y, s_sw[i_leg]),
                                polyval_bz(co_z, s_sw[i_leg])])
    elif FSM == 4:
        for i_leg in range(4):
            idx = 3 * i_leg + np.arange(1, 4) - 1
            co_x = np.linspace(pf_trans[idx[0]], p_cap[0, i_leg], 6)
            co_y = np.linspace(pf_trans[idx[1]], p_cap[1, i_leg], 6)
            co_z = np.array([0, 0, 0.15, 0.15, 0, -0.002])
            pfd[idx] = np.array([polyval_bz(co_x, s_sw[i_leg]),
                                polyval_bz(co_y, s_sw[i_leg]),
                                polyval_bz(co_z, s_sw[i_leg])])

    Xd[idx_pf, :] = np.tile(pfd, (p['predHorizon'], 1)).T
    Xd = np.tile(Xd[:, 0], (p['predHorizon'], 1)).T

    FSMout = FSM

    return FSMout, Xd, Ud, Xt


def fcn_FSM_pred_hor(FSM, Ta, t, p):
    Tst = p['Tst']
    Tsw = p['Tsw']
    T = Tst + Tsw
    Tair = 1/2 * (Tsw - Tst)

    tp = np.mod(t - Ta, T)

    if (FSM == 1) or (FSM == 3):
        Tnode = [Tst, Tair, Tst, Tair]
    else:
        Tnode = [Tair, Tst, Tair, Tst]

    for ii in range(4):
        if tp <= np.sum(Tnode[0:ii+1]):
            FSMout = np.mod(FSM + ii, 4)
            sout = (tp - np.sum(Tnode[0:ii])) / Tnode[ii]
            break

    return FSMout, sout