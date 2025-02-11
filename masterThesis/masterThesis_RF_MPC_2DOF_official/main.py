# Representation-Free Model Predictive Control (RF-MPC)
# Based on the open-resource code from Yanran Ding
# Original MATLAB code: https://github.com/YanranDing/RF-MPC

import math
import numpy as np
import time
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

import qpSWIFT
# A light-weight sparse quadratic programming solver
# Install from https://github.com/qpSWIFT/qpSWIFT
### Citation ###
# @article{pandala2019qpswift,
# title     = {qpSWIFT: A Real-Time Sparse Quadratic Program Solver for Robotic Applications},
# author    = {Pandala, Abhishek Goud and Ding, Yanran and Park, Hae-Won},
# journal   = {IEEE Robotics and Automation Letters},
# volume    = {4},
# number    = {4},
# pages     = {3355--3362},
# year      = {2019},
# publisher = {IEEE}
# }


### Initialization ###
import sys
sys.path.append('fcns')
sys.path.append('fcns_MPC')

from fcns.get_params import get_params
from fcns_MPC.fcn_bound_ref_traj import fcn_bound_ref_traj
from fcns.fcn_gen_XdUd import fcn_gen_XdUd
from fcns.fcn_FSM_bound import fcn_FSM_bound
from fcns.fcn_FSM import fcn_FSM
from fcns_MPC.fcn_get_QP_form_eta import fcn_get_QP_form_eta
from fcns.quadprog_py import quadprog_py
from mujoco_robot import mujoco_robot
from plot_results import plot_results

### Parameters init ###

# Gait selection
# 0-trot; 1-bound; 2-pacing 3-gallop; 4-trot run; 5-crawl
gait = 1

p = get_params(gait)
use_qpSWIFT = 1 # 0 - scipy.optimize.minimize(); 1 - qpSWIFT (need installation)

dt_sim = p['simTimeStep']
SimTimeDuration = 3  # [sec]
MAX_ITER = math.floor(SimTimeDuration/p['simTimeStep'])

# Desired trajectory
# A stabil setting for bounding [1]: acc=1; vel=(0.5, 0), yaw_d=0
# A stabil setting for walk trotting [0]: acc=1; vel=(0.2, 0), yaw_d=0
# Other gaits have not been tested!
p['acc_d'] = 1
p['vel_d'] = np.array([0.5, 0])
p['yaw_d'] = 0

### Model Predictive Control ###

# Initial condition
# Xt = [pc(3) dpc(3) vR(9) wb(3) pf(12)]': (30, )
if gait == 1:
    [p,Xt,Ut] = fcn_bound_ref_traj(p)
else:
    [Xt,Ut] = fcn_gen_XdUd([0,], [], np.array([[1,],[1,],[1,],[1,]]), p)
# print('p=', p)
# print('Xt=', Xt)
# print('Ut=', Ut)

# Data record
Plot_List = plot_results()

# Robot Class init
robot = mujoco_robot(p)
Xt = robot.idle_motion()

# Simulation
tsimstart = time.time()

# for ii in range(1):
for ii in range(MAX_ITER):
    print("--------------------------------------------------------------")
    print('Iteration', ii+1, 'start!')

    # Time vector
    t_ = dt_sim * ii + p['Tmpc'] * np.arange(p['predHorizon'])

    # FSM (Finite State Machine)
    if gait == 1:
        [FSM, Xd, Ud, Xt] = fcn_FSM_bound(t_, Xt, p)
    else:
        [FSM, Xd, Ud, Xt] = fcn_FSM(t_, Xt, p)

    # Form QP (Quadratic programming problem)
    [H, g, Aineq, bineq, Aeq, beq] = fcn_get_QP_form_eta(Xt, Ut, Xd, Ud, p)

    # Solve QP
    # Only qpSWIFT is able to achieve real-time control !!!
    if not use_qpSWIFT:
        # Solve QP here using scipy.optimize.minimize()
        # Unlike "quadprog" in Matlab, 
        # ...minimize() has more powerful function, 
        # but need more specific settings
        zval = quadprog_py(H, g, Aineq, bineq, Aeq, beq)
    else:
        # Interface with the QP solver "qpSWIFT"
        # Compare with scipy.optimize.minimize(), which cost c.a. 0.5-0.6 sec for one iteration
        # qpSWIFT need just c.a. 0.01-0.02 sec for one iteration
        zval = qpSWIFT.run(g, bineq, H, Aineq, Aeq, beq)['sol']
    
    # Desired GRF of four legs in 3 coordinates
    Ut = Ut + zval[0:12] 
    # print("Xt=", Xt)
    # print("Xd=", Xd[:, 0])
    # print("Ut=", Ut)

    # Do simulation in Mujoco
    ctrlData = robot.computeTorque(FSM, Xt, Ut, Xd, printResult=False) # compute torque based on desired GRF and swing leg trajectory
    u_real = robot.step(ctrlData, printGRF=False)
    Xt = robot.get_state()

    # Record results
    Plot_List.update(ii/MAX_ITER*SimTimeDuration, Xt, Xd[:, 0], Ut, u_real)
    
tsimend = time.time()
print('The total simulation time is', tsimend - tsimstart, 'sec.')

# Plot results
Plot_List.result_plot()