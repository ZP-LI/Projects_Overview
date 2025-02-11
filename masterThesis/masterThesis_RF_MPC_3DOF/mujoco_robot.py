# This Class used for interaction between MPC and MuJoCo Env 
# Include the functions like "state estimator", "idle motion", "leg control" u.s.w.

from mujoco_py import load_model_from_path, MjSim, MjViewer

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from scipy.linalg import expm
from fcns.hatMap import hatMap
from fcns.rx import rx
from fcns.ry import ry
from fcns.rz import rz
from fcns.quaternion_to_euler import quaternion_to_euler

class mujoco_robot:
    def __init__(self, p):
        self.gait = p['gait']
        self.L = p['L']                      # body length
        self.W = p['W']                      # body width
        self.d = p['d']                      # ABAD offset
        self.h = p['h']                      # body height
        self.l1 = p['l1']
        self.l2 = p['l2']
        self.g = p['g']
        self.Kp = p['Kp_sw']
        self.Kd = p['Kd_sw']
        self.Kp_init = p['Kp_init']
        self.Kd_init = p['Kd_init']
        self.timestep = p['simTimeStep']
        self.z0 = p['z0']                    # nominal COM height
        self.hipPos = p['pf34']              # 4 hip position related to COM position (!! under body frame !!)

        ## Contact detect parameter ##
        self.IsContact_Count = np.zeros((4, ))

        # Leg Index (sign of Leg/Hip direction in body length and width direction)
        self.legIndex = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) # Joint Position [FL, FR, RL, RR]

        ## Mujoco Env basic setting ##
        self.model = load_model_from_path("models/model_origin.xml")
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        # - Sensordata: 
        # - [0:2] - COM Angle Velocity [x,y,z]
        # - [3:5] - COM Linear Acceleration under body frame [x,y,z]
        # - [6:8] - [FL]*[HipMotor,UpMotor,DownMotor]
        # - [9:11] - [FR]*[HipMotor,UpMotor,DownMotor]
        # - [12:14] - [RL]*[HipMotor,UpMotor,DownMotor]
        # - [15:17] - [RR]*[HipMotor,UpMotor,DownMotor]
        # - [18:20] - COM Linear Position under world frame [x,y,z]
        # - [21:23] - COM Linear Velocity under world frame [x,y,z]
        # - [24:26] - COM Linear Acceleration under world frame [x,y,z]
        # - [27:30] - COM quaternion under world frame
        # - [31:34] - Foot Contact Detection [FL,FR,RL,RR]

        # Init Viewing Angle Setting and running speed of simulation
        self.viewer.cam.azimuth = 270
        self.viewer.cam.elevation = -20
        self.viewer.cam.lookat[0] += 1.0
        self.viewer.cam.lookat[1] += -0
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer._run_speed = 0.2

        self.sim_state = self.sim.get_state()
        self.sim.set_state(self.sim_state)

        # Robot parameter initialization ##
        self.set_init_state()

    def set_init_state(self):
        ## Related to Main Body COM ##
        self.AngPos = np.zeros((3, ))      # Init Orientation of the robot main body in form of euler angle (!! under world frame !!)
        self.R = expm(hatMap(self.AngPos)) # Rotation matrix from body frame to world frame
        self.AngVel = np.zeros((3, ))      # Init Angle Velocity of the robot main body (!! under body frame !!)
        self.AngVel_pre = self.AngVel      # Init Angle Velocity in last timestep, used to approximate COM Angle Position

        self.LinPos = np.zeros((3, ))      # Init main body COM position set in xml-file (!! under world frame !!)
        self.LinPos[-1] = 0.23 
        self.LinVel = np.zeros((3, ))      # Init main body COM linear velocity set in xml-file (!! under world frame !!)
        self.LinVel_pre = self.LinVel      # COM velocity in last timestep, used to approximate COM linear Position
        self.LinAcc = np.zeros((3, ))      # Init main body COM linear acceleration set in xml-file (!! under world frame !!)
        self.LinAcc[-1] += - 9.81          # gravity acceleration in z-axis
        self.LinAcc_pre = self.LinAcc      # COM acceletation in last timestep, used to approximate COM linear velocity

        ## Related to Leg ##
        self.Ang = np.zeros((12, ))        # Init angle of leg joints (actuator length)
        # self.Ang = self.sim.data.sensordata[6:18] # Get the init angle from sensordata setting in xml-files
        for i in range(4):
            self.Ang[i*3 + 0] = self.sim.data.sensordata[6 + i*3 + 0] + 0 /180*math.pi
            self.Ang[i*3 + 1] = self.sim.data.sensordata[6 + i*3 + 1] + 135 /180*math.pi
            self.Ang[i*3 + 2] = self.sim.data.sensordata[6 + i*3 + 2] - 90 /180*math.pi
        self.Ang_pre = np.zeros((12, ))    # Angle of leg joints in last timestep, used for PD-Control
        for i in range(len(self.Ang)):
            self.Ang_pre[i] = self.Ang[i]
        
        ## Related to Foot ##
        self.footPos = np.zeros((12,)) # init foot position (!! under leg frame !!)
        self.footPos_pre = np.zeros((12,))
        self.footPos = self.forward_kin_3d()
        for i in range(len(self.footPos)):
            self.footPos_pre[i] = self.footPos[i]
    
    def idle_motion(self):
        # To get a stable initial robot state
        for loop in range(500):
            ctrlData = np.zeros((12, ))

            for i in range(4):
                ang = self.Ang[i*3 + 0 : i*3 + 3]
                ang_pre = self.Ang_pre[i*3 + 0 : i*3 + 3]
                sign_W = self.legIndex[i, 1]

                # Current foot position related to respectiv hip joint
                footPos_Cur = self.forward_kin_h2f(ang, sign_W)
                footPos_Pre = self.forward_kin_h2f(ang_pre, sign_W)
                footPos_Vel = (footPos_Cur - footPos_Pre) / self.timestep

                # Desired foot postion related to the corresponding hip joint
                footPos_des = np.array([0, sign_W * self.d, - 0.2 - 0.030]) # Plus 0.05 of z-coordinate to offset the gravity influence of VMC
                # print(footPos_des)

                # Use PD-control to control leg motion to follow the reference swing leg trajectory
                # Virtuell Machine Control - VMC
                Kp = self.Kp_init
                Kd = self.Kd_init
                footPos_err = Kp * (footPos_des - footPos_Cur) + Kd * (0 - footPos_Vel)

                # Use Jacobi Matrix that computed from kinematic analysis 
                # to compute the desired output torque of every motor (DOF) on the leg
                JT = self.foot_jacobi(ang, sign_W)
                tau = JT @ footPos_err

                ctrlData[i*3: i*3+3] = tau
            
            self.step(ctrlData)
            Xt = self.get_state()
        
        return Xt

    def step(self, ctrlData):
        ## Transfer computed torque to Mujoco-Env and rendering
        self.sim.data.ctrl[:] = ctrlData
        self.sim.step()
        self.viewer.render()

        touch_GRF = np.around(self.sim.data.sensordata[31:35], decimals=4) # GRF [z]
        full_GRF = np.around(self.sim.data.efc_force[0:12], decimals=4) # GRF [z,y,x]*4
        print("touch sensor:", touch_GRF)
        print("Ground Reaction Force:", full_GRF)
        u_real = np.zeros_like(full_GRF)
        for i in range(len(touch_GRF)):
            if np.absolute(touch_GRF[i] - 0) > 1e-4:
                Fz_index = np.where(touch_GRF[i] == full_GRF)
                for ii in range(len(Fz_index[0])):
                    if Fz_index[0][ii] % 3 == 0:
                        u_real[3*i : 3*i+3] = full_GRF[Fz_index[0][ii] : Fz_index[0][ii]+3][::-1]
                        break
        
        return u_real # Return the real GRFs with the same structure as Ut
 
    def get_state(self):
        ## Related to Main Body COM ##
        # True COM Angle Position under world frame #
        self.AngVel_pre = self.AngVel
        Quat = self.sim.data.sensordata[27:31]
        self.AngPos = quaternion_to_euler(Quat)
        self.R = expm(hatMap(self.AngPos)) # Rotation matrix
        # print("COM Angle Position in world frame:", np.around(AngPos_w, decimals=4))

        # Test of COM Angle Position Measurement #
        # self.AngPos += (self.R @ self.AngVel + self.R @ self.AngVel_pre) * self.timestep / 2 # COM global euler angle is computed from COM angle velocity under body frame
        # self.R = expm(hatMap(self.AngPos)) # Rotation matrix
        # self.AngVel = self.sim.data.sensordata[0:3] # COM local velocity can be taken direktly from Gyro Sensor
        # self.AngVel_pre = self.AngVel
        # print("COM Angle Velocity:", np.around(self.AngVel, decimals=4), "COM Angle Position:", np.around(self.AngPos, decimals=4))

        # True COM Linear Position/Velocity under world frame #
        Ref_LinPos = self.sim.data.sensordata[18:21]
        Ref_LinVel = self.sim.data.sensordata[21:24]
        self.LinPos = Ref_LinPos
        self.LinVel = Ref_LinVel
        # Ref_LinAcc = self.sim.data.sensordata[24:27]
        # Ref_LinAcc[-1] += - 9.81

        # Test of COM Linear Position/Velocity Measurement #
        # self.LinAcc = self.sim.data.sensordata[3:6]
        # self.LinAcc = self.R @ self.LinAcc
        # self.LinAcc[-1] += - 9.81
        # self.LinVel += (self.LinAcc + self.LinAcc_pre) * self.timestep / 2
        # self.LinAcc_pre = self.LinAcc
        # self.LinPos += self.LinVel_pre * self.timestep + 1/2 * self.LinAcc * self.timestep**2
        # self.LinVel_pre = self.LinVel
        # print('------------------------------------------------------')
        # print("Real COM Linear Position:", np.around(Ref_LinPos, decimals=4))
        # print("COM Linear Position:", np.around(self.LinPos, decimals=4))
        # print('------')
        # print("Real COM Linear Velocity:", np.around(Ref_LinVel, decimals=4))
        # print("COM Linear Velocity:", np.around(self.LinVel, decimals=4))
        # print('------')
        # print("Real COM Linear Acceleration:", np.around(Ref_LinAcc, decimals=4))
        # print("COM Linear Acceleration:", np.around(self.LinAcc, decimals=4))

        ## Related to Leg ##
        l1 = self.l1
        l2 = self.l2
        for i in range(len(self.Ang)):
            self.Ang_pre[i] = self.Ang[i]
        for i in range(4):
            self.Ang[i*3 + 0] = self.sim.data.sensordata[6 + i*3 + 0] + 0 /180*math.pi
            self.Ang[i*3 + 1] = self.sim.data.sensordata[6 + i*3 + 1] + 135 /180*math.pi
            self.Ang[i*3 + 2] = self.sim.data.sensordata[6 + i*3 + 2] - 90 /180*math.pi

        ## Related to Foot ##
        for i in range(len(self.footPos)):
            self.footPos_pre[i] = self.footPos[i]
        self.footPos = self.forward_kin_3d()
        
        Xt = np.concatenate((self.LinPos, self.LinVel, self.R.flatten('F'), self.AngVel, self.footPos))
            
        return Xt

    def computeTorque(self, FSM, Xt, Ut, Xd):
        # Force Control: Compute leg torques based on desired GRF and Foot Jacobi Matrix
        # Swing Leg Control: Compute leg torques based on reference swing leg trajectory (desired foot position)

        if self.gait == 1: # Bound
            # leg_ID: 0, 1, 2, 3 for FL, FR, RL, RR
            # FSM - 1: Front Leg - Force Control; Rear Leg - Swing Leg Control
            # FSM - 2: Front Leg - Swing Leg Control; Rear Leg - Swing Leg Control
            # FSM - 3: Front Leg - Swing Leg Control; Rear Leg - Force Control
            # FSM - 4: Front Leg - Swing Leg Control; Rear Leg - Swing Leg Control
            if FSM == 1:
                tau_FL = self.force_control(1, Xt, Ut) # + self.force_control_compensate(1)
                tau_FR = self.force_control(2, Xt, Ut) # + self.force_control_compensate(2)
                tau_RL = self.swing_leg_control(3, Xt, Xd)
                tau_RR = self.swing_leg_control(4, Xt, Xd)
            elif FSM == 2:
                tau_FL = self.swing_leg_control(1, Xt, Xd)
                tau_FR = self.swing_leg_control(2, Xt, Xd)
                tau_RL = self.swing_leg_control(3, Xt, Xd)
                tau_RR = self.swing_leg_control(4, Xt, Xd)
            elif FSM == 3:
                tau_FL = self.swing_leg_control(1, Xt, Xd)
                tau_FR = self.swing_leg_control(2, Xt, Xd)
                tau_RL = self.force_control(3, Xt, Ut)  # + self.force_control_compensate(3)
                tau_RR = self.force_control(4, Xt, Ut)  # + self.force_control_compensate(4)
            elif FSM == 4:
                tau_FL = self.swing_leg_control(1, Xt, Xd)
                tau_FR = self.swing_leg_control(2, Xt, Xd)
                tau_RL = self.swing_leg_control(3, Xt, Xd)
                tau_RR = self.swing_leg_control(4, Xt, Xd)
            CtrlData = np.concatenate((tau_FL, tau_FR, tau_RL, tau_RR))
        else:
            # FSM[i] - 1: Leg in stance phase
            # FSM[i] - 2: Leg in swing phase
            CtrlData = np.zeros((12, ))
            for i in range(4):
                if FSM[i] == 1: # stance phase
                    tau = self.force_control(i+1, Xt, Ut)
                elif FSM[i] == 2: # swing phase
                    tau = self.swing_leg_control(i+1, Xd)
                CtrlData[[i*3+0, i*3+1, i*3+2]] = tau

        return CtrlData

    def force_control(self, leg_ID, Xt, Ut):
        # leg_ID: 0, 1, 2, 3 for FL, FR, RL, RR
        # Ang: Angle of rotation of every actuator/motor (actuator length)
        #      (12, ): [FL_actuator - 1,2,3, ..., RR_actuator - 1,2,3]

        # Reset count of contact detection of this leg
        self.IsContact_Count[leg_ID - 1] = 0

        # Get leg joint angle
        ang = self.Ang[(leg_ID - 1)*3 + 0 : (leg_ID - 1)*3 + 3]

        # Get leg index
        sign_W = self.legIndex[leg_ID - 1, 1]

        # Computer desired GRFs (Ground reaction force) in 3-coordinate under body frame
        GRF_coef = 1
        GRF_w = np.array([Ut[(leg_ID - 1) * 3 + 0], 
                          - Ut[(leg_ID - 1) * 3 + 1], 
                          - Ut[(leg_ID - 1) * 3 + 2]])
        R = np.reshape(Xt[6:15], (3, 3), order='F')
        GRF_b = R.T @ GRF_w
        F = GRF_b * GRF_coef

        # Use Jacobi Matrix computed from kinematic analysis 
        # to compute the desired output torque of every motor (DOF) on the leg
        JT = self.foot_jacobi(ang, sign_W)
        tau = JT @ F

        return tau # check later!!!

    def swing_leg_control(self, leg_ID, Xt, Xd):
        # Gain value of PD-Control
        Kp = self.Kp
        Kd = self.Kd

        # Leg joint angle position/velocity
        ang = self.Ang[(leg_ID - 1)*3 + 0 : (leg_ID - 1)*3 + 3]
        ang_pre = self.Ang_pre[(leg_ID - 1)*3 + 0 : (leg_ID - 1)*3 + 3]
        ang_vel = (ang - ang_pre) / self.timestep

        # Contact detection
        if self.sim.data.sensordata[23 + leg_ID - 1] > 1e-2:
            self.IsContact_Count[leg_ID - 1] += 1
        else:
            self.IsContact_Count[leg_ID - 1] = 0
        
        # If the foot touches the ground earlier than the plan,
        # then maintain the leg posture since the timestep of contact, 
        # no longer follow the swing leg trajectory to prevent the foot from bouncing,
        # and waiting for the next stance phase of this leg
        if self.IsContact_Count[leg_ID - 1] >= 1:
            print(leg_ID, "Contact detected!")
            Kp = 0.01 * Kp
            Kd = 0.01 * Kd

        # Calculate the desired joint angle based on the desired foot position
        footPos_des_w = np.array([Xd[18 + (leg_ID - 1) * 3 + 0, 0], 
                                  Xd[18 + (leg_ID - 1) * 3 + 1, 0], 
                                  Xd[18 + (leg_ID - 1) * 3 + 2, 0]])
        sign_L = self.legIndex[leg_ID - 1, 0]
        sign_W = self.legIndex[leg_ID - 1, 1]
        ang_des = self.inverse_kin_3d(footPos_des_w, sign_L, sign_W, Xd)

        # Use PD-control to control leg motion to follow the reference swing leg trajectory
        tau = Kp * (ang_des - ang) + Kd * (0 - ang_vel)

        return tau
    
    def force_control_compensate(self, leg_ID):
        # Gain value of PD-Control
        Kp = self.Kp
        Kd = self.Kd

        # Hip Leg joint angle position/velocity
        ang = self.Ang[(leg_ID - 1)*3 + 0]
        ang_pre = self.Ang_pre[(leg_ID - 1)*3 + 0]
        ang_vel = (ang - ang_pre) / self.timestep

        # Use PD-control to control leg motion to follow the reference swing leg trajectory
        tau = np.zeros(3, )
        tau[0] = Kp[0] * (0 - ang) + Kd[0] * (0 - ang_vel)

        print("current hip joint position:", ang, ", Velocity:", ang_vel, ", PD torque:", tau)

        return tau
    
    def forward_kin_3d(self):
        # Based on current leg joint angle and COM position
        # computer the foot position under world frame
        L = self.L
        W = self.W
        d = self.d
        l1 = self.l1
        l2 = self.l2
        pf_w = np.zeros((12, )) # Foot position under world frame

        for i in range(4):
            sign_L = self.legIndex[i, 0]
            sign_W = self.legIndex[i, 1]

            T_w_2_com = np.vstack((np.hstack((self.R, self.LinPos.reshape(3, 1))), 
                                   np.array([[0, 0, 0, 1]])))
            T_com_2_h = np.vstack((np.hstack((rx(self.Ang[3*i+0]), np.array([[sign_L*L/2], [sign_W*W/2], [0]]))), 
                                   np.array([[0, 0, 0, 1]])))
            T_h_2_s = np.vstack((np.hstack((ry(self.Ang[3*i+1]), np.array([[0], [sign_W*d], [0]]))), 
                                   np.array([[0, 0, 0, 1]])))
            T_s_2_k = np.vstack((np.hstack((ry(self.Ang[3*i+2]), np.array([[l1], [0], [0]]))), 
                                   np.array([[0, 0, 0, 1]])))
            T_k_2_f = np.vstack((np.hstack((np.eye(3), np.array([[l2], [0], [0]]))), 
                                   np.array([[0, 0, 0, 1]])))
            T_w_2_f = T_w_2_com @ T_com_2_h @ T_h_2_s @ T_s_2_k @ T_k_2_f

            pf_w[3*i+0 : 3*i+3] = T_w_2_f[[0, 1, 2], -1]
        
        
        return pf_w
    
    def forward_kin_h2f(self, Ang, sign_W):
        L = self.L
        W = self.W
        d = self.d
        l1 = self.l1
        l2 = self.l2

        T_h_2_h = np.vstack((np.hstack((rx(Ang[0]), np.array([[0], [0], [0]]))), 
                                   np.array([[0, 0, 0, 1]])))
        T_h_2_s = np.vstack((np.hstack((ry(Ang[1]), np.array([[0], [sign_W*d], [0]]))), 
                                np.array([[0, 0, 0, 1]])))
        T_s_2_k = np.vstack((np.hstack((ry(Ang[2]), np.array([[l1], [0], [0]]))), 
                                np.array([[0, 0, 0, 1]])))
        T_k_2_f = np.vstack((np.hstack((np.eye(3), np.array([[l2], [0], [0]]))), 
                                np.array([[0, 0, 0, 1]])))
        T_h_2_f = T_h_2_h @ T_h_2_s @ T_s_2_k @ T_k_2_f

        pf_h2f = T_h_2_f[[0, 1, 2], -1]
        
        return pf_h2f
    
    def foot_jacobi(self, ang, sign_W):
        l1 = self.l1
        l2 = self.l2
        d = self.d
        q1 = ang[0]
        q2 = ang[1]
        q3 = ang[2]

        j11 = 0
        j12 = - l2 * (math.cos(q2)*math.sin(q3) + math.sin(q2)*math.cos(q3)) - l1 * math.sin(q2)
        j13 = - l2 * (math.cos(q2)*math.sin(q3) + math.sin(q2)*math.cos(q3))
        j21 = l1 * math.cos(q1)*math.sin(q2) + l2 * (math.cos(q1)*math.cos(q2)*math.sin(q3) + math.cos(q1)*math.sin(q2)*math.cos(q3)) - sign_W * d * math.sin(q1)
        j22 = l1 * math.sin(q1)*math.cos(q2) - l2 * (math.sin(q1)*math.sin(q2)*math.sin(q3) - math.sin(q1)*math.cos(q2)*math.cos(q3))
        j23 = - l2 * (math.sin(q1)*math.sin(q2)*math.sin(q3) - math.sin(q1)*math.cos(q2)*math.cos(q3))
        j31 = l1 * math.sin(q1)*math.sin(q2) + l2 * (math.sin(q1)*math.cos(q2)*math.sin(q3) + math.sin(q1)*math.sin(q2)*math.cos(q3)) - sign_W * d * math.cos(q1)
        j32 = - l1 * math.cos(q1)*math.cos(q2) - l2 * (math.cos(q1)*math.cos(q2)*math.cos(q3) - math.cos(q1)*math.sin(q2)*math.sin(q3))
        j33 = - l2 * (math.cos(q1)*math.cos(q2)*math.cos(q3) - math.cos(q1)*math.sin(q2)*math.sin(q3))

        J = np.array([[j11, j12, j13],
                      [j21, j22, j23],
                      [j31, j32, j33]])
        JT = np.transpose(J)

        return JT
    
    def inverse_kin_3d(self, p_f_w, sign_L, sign_W, Xd):
        # Robot state
        pcom = np.array([Xd[0:3, 0]])                  # Desired COM position under world frame
        R = np.reshape(Xd[6:15, 0], (3, 3), order='F') # Desired rotation matrix

        # Parameters
        L = self.L
        W = self.W
        l1 = self.l1
        l2 = self.l2
        d = self.d

        # Convert the desired foot position under world frame
        # to the relativ position between hip joint and foot under body frame
        T_w_2_com = np.vstack((np.hstack((R, pcom.reshape(3, 1))), 
                                   np.array([[0, 0, 0, 1]])))
        T_com_2_h = np.vstack((np.hstack((np.eye(3), np.array([[sign_L*L/2], [sign_W*W/2], [0]]))), 
                                   np.array([[0, 0, 0, 1]])))
        T_w_2_h = T_w_2_com @ T_com_2_h
        p_h_w = T_w_2_h[[0, 1, 2], -1]       # Hip joint position under world frame
        p_h2f_w = p_f_w - p_h_w              # Relativ position between hip joint and foot under world frame
        p_h2f_b = np.transpose(R) @ p_h2f_w  # Relativ position between hip joint and foot under body frame

        # Main part of inverse kinematic
        # q1
        p_f_yz = p_h2f_b[1:3]
        a = np.linalg.norm(p_f_yz)
        alpha = math.asin(p_h2f_b[1] / a)
        belta = math.asin(d / a)
        q1 = alpha - sign_W * belta

        # q2
        Line_HG = sign_W * np.array([0, d * math.cos(q1), d * math.sin(q1)])
        b = np.linalg.norm(p_h2f_b - Line_HG)
        p_f_xz = np.transpose(rx(q1)) @ (p_h2f_b - Line_HG)
        if p_f_xz[2] <= 0:
            gamma = math.acos(p_f_xz[0] / b)
        else:
            if p_f_xz[0] >= 0:
                gamma = - math.asin(p_f_xz[2] / b)
            else:
                gamma = np.pi + math.asin(p_f_xz[2] / b)
        cos_psi = (l1**2 + b**2 - l2**2) / (2 * l1 * b)
        psi = math.acos(cos_psi)
        q2 = gamma + psi
        
        # q3
        cos_theta = (l1**2 + l2**2 - b**2) / (2 * l1 * l2)
        theta = math.acos(cos_theta)
        q3 = - (np.pi - theta)

        q = np.array([q1, q2, q3])

        return q
    
    def inverse_kin_h2f(self, p_h2f_b, sign_W):
        # Parameters
        l1 = self.l1
        l2 = self.l2
        d = self.d

        # Main part of inverse kinematic
        # q1
        p_f_yz = p_h2f_b[1:3]
        a = np.linalg.norm(p_f_yz)
        alpha = math.asin(p_h2f_b[1] / a)
        belta = math.asin(d / a)
        q1 = alpha - sign_W * belta

        # q2
        Line_HG = sign_W * np.array([0, d * math.cos(q1), d * math.sin(q1)])
        b = np.linalg.norm(p_h2f_b - Line_HG)
        p_f_xz = np.transpose(rx(q1)) @ (p_h2f_b - Line_HG)
        if p_f_xz[2] <= 0:
            gamma = math.acos(p_f_xz[0] / b)
        else:
            if p_f_xz[0] >= 0:
                gamma = - math.asin(p_f_xz[2] / b)
            else:
                gamma = np.pi + math.asin(p_f_xz[2] / b)
        cos_psi = (l1**2 + b**2 - l2**2) / (2 * l1 * b)
        psi = math.acos(cos_psi)
        q2 = gamma + psi
        
        # q3
        cos_theta = (l1**2 + l2**2 - b**2) / (2 * l1 * l2)
        theta = math.acos(cos_theta)
        q3 = - (np.pi - theta)

        q = np.array([q1, q2, q3])

        return q

    def circle_test(self):
        # Gain values for PD-Control
        Kp = self.Kp
        Kd = self.Kd

        # Circle target trajectory
        a = 0.04
        b = 0.02
        theta = np.linspace(90, 450, num=100)
        theta = theta /180*math.pi
        footPos_desTraj = np.zeros((len(theta), 3))
        footPos_realTraj = np.zeros_like(footPos_desTraj)
        center = np.array([0, 0, - 0.178])
        for i in range(len(theta)):
            x = a * math.cos(theta[i])
            y = a * math.cos(theta[i])
            z = - b * math.sin(theta[i])
            cur_point  = center + np.array([x, y, z])
            footPos_desTraj[i, 0:3] = cur_point

        d = self.d
        ctrlData = np.zeros((12, ))
        for loop in range(len(theta)):
            print("--------------Iteration:", loop, "--------------")
            footPos_des_general = footPos_desTraj[loop, :]
            for i in range(4):
                # i = [0,1,2,3] correspond to [FL,FR,RL,RR]
                print(footPos_des_general)

                # Leg joint angle position/velocity
                ang = self.Ang[i*3 + 0 : i*3 + 3]
                ang_pre = self.Ang_pre[i*3 + 0 : i*3 + 3]
                ang_vel = (ang - ang_pre) / self.timestep

                # Calculatethe desired joint angle based on the desired foot position
                sign_W = self.legIndex[i, 1]
                footPos_des = np.array([footPos_des_general[0],
                                        sign_W * footPos_des_general[1] + sign_W * d,
                                        footPos_des_general[2]])
                print("Leg ID:", i, "Desired foot position:", footPos_des)
                ang_des = self.inverse_kin_h2f(footPos_des, sign_W)

                # Use PD-control to control leg motion to follow the reference swing leg trajectory
                tau = Kp * (ang_des - ang) + Kd * (0 - ang_vel)
                ctrlData[i*3 + 0 : i*3 + 3] = tau

                # Computer current foot position related to hip joint
                footPos_real = self.forward_kin_h2f(ang, sign_W)
                if i == 0:
                    footPos_realTraj[loop, :] = footPos_real

                # Information printing
                # if i == 0:
                #     print("Desired Angle:", ang_des, "Real Angle:", ang)
                #     print("Previous Angle:", ang_pre)

            self.step(ctrlData)
            self.get_state()

        ## Animation for FL-Leg
        footPos_desTraj[:, 1] += 1 * d
        fig2, ax2 = plt.subplots(1, 2, figsize=(20, 5))
        ax2[0].set_xlabel("x-axis", size=20)
        ax2[0].set_ylabel("z-axis", size=20)
        ax2[1].set_xlabel("y-axis", size=20)
        ax2[1].set_ylabel("z-axis", size=20)
        ax2[0].set_xlim([-0.055, 0.055])
        ax2[1].set_xlim([-0.055 + 1*d, 0.055 + 1*d])
        line1, = ax2[0].plot(footPos_desTraj[:, 0], footPos_desTraj[:, 2], color="green", label='desired')
        line2, = ax2[0].plot(footPos_realTraj[:, 0], footPos_realTraj[:, 2], color="blue", label='real')
        line3, = ax2[1].plot(footPos_desTraj[:, 1], footPos_desTraj[:, 2], color="green", label='desired')
        line4, = ax2[1].plot(footPos_realTraj[:, 1], footPos_realTraj[:, 2], color="blue", label='real')
        fig2.legend([line1, line2], ['desired', 'real'], bbox_to_anchor=(0.75,0.99), prop={'size': 20})
        def animate(i):
            line1.set_xdata(footPos_desTraj[:, 0][0:i+1])
            line1.set_ydata(footPos_desTraj[:, 2][0:i+1])
            line2.set_xdata(footPos_realTraj[:, 0][0:i+1])
            line2.set_ydata(footPos_realTraj[:, 2][0:i+1])
            line3.set_xdata(footPos_desTraj[:, 1][0:i+1])
            line3.set_ydata(footPos_desTraj[:, 2][0:i+1])
            line4.set_xdata(footPos_realTraj[:, 1][0:i+1])
            line4.set_ydata(footPos_realTraj[:, 2][0:i+1])
            return line1, line2, line3, line4
        ani = animation.FuncAnimation(
            fig2, animate, interval=20, blit=True, save_count=100)

        ani.save("simple_animation.gif", dpi=300,
                writer=PillowWriter(fps=5))
        plt.show()