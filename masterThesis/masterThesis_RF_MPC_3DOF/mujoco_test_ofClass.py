import math
import numpy as np
import time
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

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
# from fcns.computeTorque import computeTorque
from mujoco_robot import mujoco_robot

### Parameters init ###

# Gait selection
# 0-trot; 1-bound; 2-pacing 3-gallop; 4-trot run; 5-crawl
gait = 1

p = get_params(gait)
use_qpSWIFT = 1 # 0 - scipy.optimize.minimize(), 1 - qpSWIFT (need installation)

dt_sim = p['simTimeStep']
SimTimeDuration = 0.5  # [sec]
MAX_ITER = math.floor(SimTimeDuration/p['simTimeStep'])

# Desired trajectory
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

# Robot Class init
robot = mujoco_robot(p)
# robot.idle_motion()
# robot.circle_test()

# GRF stand tes
'''
for loop in range(500):
    ctrldata = np.zeros(12, )
    for i in range(4):
        # Get leg joint angle
        ang = robot.Ang[i*3 + 0 : i*3 + 3]

        # Get leg index
        sign_W = robot.legIndex[i, 1]

        # Computer desired GRFs (Ground reaction force) in 3-coordinate under body frame
        GRF_coef = 1
        GRF_w = np.array([0, -5, - 15.])
        R = np.eye(3)
        GRF_b = R.T @ GRF_w
        F = GRF_b * GRF_coef

        # Use Jacobi Matrix computed from kinematic analysis 
        # to compute the desired output torque of every motor (DOF) on the leg
        JT = robot.foot_jacobi(ang, sign_W)
        tau = JT @ F

        ctrldata[i*3 + 0 : i*3 + 3] = tau

    u_real = robot.step(ctrldata)
    Xt = robot.get_state()
'''