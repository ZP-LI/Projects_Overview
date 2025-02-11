import numpy as np
import math

def computeTorque(p, FSM, Ang, Xt, Ut, Xd):
    # Force Control: Compute leg torques based on desired GRF and Jaccobi Matrix
    # Swing Leg Control: Compute leg torques based on reference swing leg trajectory (desired leg position)
    
    # FSM - 1: Front Leg - Force Control; Rear Leg - Swing Leg Control
    # FSM - 2: Front Leg - Swing Leg Control; Rear Leg - Swing Leg Control
    # FSM - 3: Front Leg - Swing Leg Control; Rear Leg - Force Control
    # FSM - 4: Front Leg - Swing Leg Control; Rear Leg - Swing Leg Control
    if FSM == 1:
        tau_FL = force_control(p, 1, Ang, Xt, Ut)
        tau_FR = force_control(p, 2, Ang, Xt, Ut)
        tau_RL = swing_leg_control(p, 3, Ang, Xd)
        tau_RR = swing_leg_control(p, 4, Ang, Xd)
    elif FSM == 2:
        tau_FL = swing_leg_control(p, 1, Ang, Xd)
        tau_FR = swing_leg_control(p, 2, Ang, Xd)
        tau_RL = swing_leg_control(p, 3, Ang, Xd)
        tau_RR = swing_leg_control(p, 4, Ang, Xd)
    elif FSM == 3:
        tau_FL = swing_leg_control(p, 1, Ang, Xd)
        tau_FR = swing_leg_control(p, 2, Ang, Xd)
        tau_RL = force_control(p, 3, Ang, Xt, Ut)
        tau_RR = force_control(p, 4, Ang, Xt, Ut)
    elif FSM == 4:
        tau_FL = swing_leg_control(p, 1, Ang, Xd)
        tau_FR = swing_leg_control(p, 2, Ang, Xd)
        tau_RL = swing_leg_control(p, 3, Ang, Xd)
        tau_RR = swing_leg_control(p, 4, Ang, Xd)
    
    Torques = np.concatenate((tau_FL, tau_FR, tau_RL, tau_RR))

    return Torques

def force_control(p, leg_ID, Ang, Xt, Ut):
    # leg_ID: 1, 2, 3, 4 for FL, FR, RL, RR
    # Ang: Angle of rotation of every actuator/motor (actuator length)
    #      (8, ): [FL_act1, FL_act2, FR_act1, ..., RR_act1, RR_act2]

    l1 = p['l1']
    l2 = p['l2']

    ang1 = Ang[(leg_ID - 1) * 2]
    ang2 = Ang[(leg_ID - 1) * 2 + 1]

    # Get desired GRFs (Ground reaction force) in 3-coordinate under body frame
    # For straight forward motion, there are only GRFs in x-, z- direction
    # -> Can be extended to motion with yaw-angle, but may need 3 DOFs on every leg
    GRF_w = np.array([Ut[(leg_ID - 1) * 3], 
                      Ut[(leg_ID - 1) * 3 + 1], 
                      Ut[(leg_ID - 1) * 3 + 2]])
    R = np.reshape(Xt[6:15], (3, 3), order='F')
    GRF_b = R.T @ GRF_w
    Fx = GRF_b[0]
    Fz = GRF_b[2]

    # Use Jacobi Matrix computed from kinematic analysis 
    # to compute the desired output torque of every motor (DOF) on the leg
    # based on the desired GRF computed by MPC 
    JT = np.array([[- l1*math.sin(ang1) - l2*math.sin(ang1+ang2), l1*math.cos(ang1) + l2*math.cos(ang1+ang2)], 
                    [- l1*math.sin(ang1+ang2), l2*math.cos(ang1+ang2)]])
    tau = JT @ np.array([Fx, Fz])

    return - tau

def swing_leg_control(p, leg_ID, Ang, Xd):
    
    l1 = p['l1']
    l2 = p['l2']

    # Current foot position related to the corresponding hip joint
    # Computed from the "Ang" and leg forward kinematic
    # Only position differences in x-, z- direction are considered
    ang1 = Ang[(leg_ID - 1) * 2]
    ang2 = Ang[(leg_ID - 1) * 2 + 1]
    footPos_cur = np.array([l1*math.cos(ang1) + l2*math.cos(ang1+ang2), l1*math.sin(ang1) + l2*math.sin(ang1+ang2)])

    # Desired foot postion related to the corresponding hip joint
    footPos_des_com = np.array([Xd[18 + (leg_ID - 1) * 3, 0], Xd[18 + (leg_ID - 1) * 3 + 2, 0]])
    hip_com = p['pf34'][[0, 2], leg_ID - 1]
    footPos_des = np.array([- footPos_des_com[0] + hip_com[0], p['z0'] - hip_com[1]])
    # print(footPos_des)

    # Try 1: Use only p-control to control leg motion 
    # to follow the reference swing leg trajectory
    Kp_sw = p['Kp_sw']
    footPos_err = Kp_sw * (footPos_des - footPos_cur)

    # Use Jacobi Matrix computed from kinematic analysis 
    # to compute the desired output torque of every motor (DOF) on the leg
    # based on the desired GRF computed by MPC 
    JT = np.array([[- l1*math.sin(ang1) - l2*math.sin(ang1+ang2), l1*math.cos(ang1) + l2*math.cos(ang1+ang2)], 
                    [- l1*math.sin(ang1+ang2), l2*math.cos(ang1+ang2)]])
    tau = JT @ footPos_err

    return - tau