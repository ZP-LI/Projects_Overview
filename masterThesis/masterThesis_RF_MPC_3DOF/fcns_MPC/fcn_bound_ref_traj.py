import numpy as np
from scipy.linalg import expm
from fcns.hatMap import hatMap
from fcns.bz_int import bz_int

def fcn_bound_ref_traj(p):
    # This function finds the initial condition for periodic bounding
    # The calculation is based on the paper (citation):
    # Park, Hae-Won, Patrick M. Wensing, and Sangbae Kim. 
    # "High-speed bounding with the MIT Cheetah 2: Control design and experiments."
    # The International Journal of Robotics Research 36, no. 2 (2017): 167-192.
    
    mass, J, g, Tst, Tsw = p['mass'], p['J'], p['g'], p['Tst'], p['Tsw']
    T = Tst + Tsw
    Tair = 1/2 * (Tsw - Tst)

    b_co = np.array([0, 0.8, 1, 1, 0.8, 0]) # Bezier coefficient
    b_ = np.mean(b_co)

    ## Fz ##
    # 2 * alpha * b_ * Tst = mass * g * T
    alpha_z = (mass * g * T) / (2 * b_ * Tst)
    Fz_co = alpha_z * b_co

    dz_co = bz_int(Fz_co/mass-g, 0, Tst)
    z_co = bz_int(dz_co, 0, Tst)

    # first principle: integration
    dz0 = -1 / (Tst+Tair) * (z_co[-1] + Tair*(dz_co[-1]+g*Tst)-1/2*g*((Tst+Tair)**2-Tst**2))

    dz_co = bz_int(Fz_co/mass-g, dz0, Tst)
    z_co = bz_int(dz_co, p['z0'], Tst)

    ## theta ##
    alpha_th = 180 * J[1, 1] # ???
    tau_co = - alpha_th * b_co

    dth_co = bz_int(tau_co/J[1, 1], 0, Tst)

    # by symmetry
    dth0 = -1/2 * dth_co[-1]

    th0 = dth0*Tair/2
    dth_co = bz_int(tau_co/J[1, 1], dth0, Tst)
    th_co = bz_int(dth_co, th0, Tst)

    ## output B-spline coefficient ##
    p['Fz_co'] = Fz_co
    p['dz_co'] = dz_co
    p['z_co'] = z_co

    p['tau_co'] = tau_co
    p['dth_co'] = dth_co
    p['th_co'] = th_co

    ## intial condition ##
    R0 = expm(hatMap([0, th0, 0]))
    Xt = np.concatenate(([0], [0], [p['z0']], [0], [0], [dz0], R0.flatten('F'), [0], [dth0], [0]))
    Xt = np.concatenate((Xt, p['pf34'].flatten('F')))
    Ut = np.squeeze(np.tile(np.array([[0, 0, 1/4*p['mass']*p['g']]]).T, (4, 1)))

    return p, Xt, Ut