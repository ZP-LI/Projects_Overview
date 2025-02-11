# Load xml-model to MuJoCo
from mujoco_py import load_model_from_path, MjSim, MjViewer

# Load other useful libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas
from scipy.linalg import expm
from fcns.hatMap import hatMap

class mujoco_robot:
    def __init__(self, p):
        self.gait = p['gait']
        self.l1 = p['l1']
        self.l2 = p['l2']
        self.g = p['g']
        self.Kp = p['Kp_sw']
        self.Kd = p['Kd_sw']
        self.Kp_init = p['Kp_init']
        self.Kd_init = p['Kd_init']
        self.timestep = p['simTimeStep']
        self.z0 = p['z0'] # nominal COM height
        self.hipPos = p['pf34'] # 4 hip position related to COM position (!! under body frame !!)

        # Contact sensor
        self.IsContact_Count = np.zeros((4, ))

        # Mujoco Env basic setting
        self.model = load_model_from_path("models/model_origin.xml")
        # - Sensordata: 
        # - [0:2] - COM Angle Velocity [x,y,z]
        # - [3:5] - COM Linear Acceleration under body frame [x,y,z]
        # - [6:7] - [FL]*[UpMotor,DownMotor]
        # - [8:9] - [FR]*[UpMotor,DownMotor]
        # - [10:11] - [RL]*[UpMotor,DownMotor]
        # - [12:13] - [RR]*[UpMotor,DownMotor]
        # - [14:16] - COM Linear Position under world frame [x,y,z]
        # - [17:19] - COM Linear Velocity under world frame [x,y,z]
        # - [20:22] - COM Linear Acceleration under world frame [x,y,z]
        # - [23:26] - Foot Contact Detection [FL,FR,RL,RR]
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        # Simulation viewer setting
        self.viewer.cam.azimuth = 270
        self.viewer.cam.elevation = -20
        self.viewer.cam.lookat[0] += 1.0
        self.viewer.cam.lookat[1] += -0
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.distance = self.model.stat.extent * 1.5

        # Initial simulation speed ("1" for real-time)
        self.viewer._run_speed = 0.2

        # Set code for MuJoCo Simulation
        self.sim_state = self.sim.get_state()
        self.sim.set_state(self.sim_state)

        # Initialization of Robot variables
        self.set_init_state()

    def set_init_state(self):
        ## Related to Main Body COM
        self.AngPos = np.zeros((3, )) # Init Orientation of the robot main body in form of euler angle (!! under world frame !!)
        self.R = expm(hatMap(self.AngPos)) # Rotation matrix from body frame to world frame
        self.AngVel = np.zeros((3, )) # Init Euler Angle Velocity of the robot main body               (!! under body frame !!)
        self.AngVel_pre = self.AngVel # Euler Angle Velocity in last timestep

        self.LinPos = np.zeros((3, )) # Init main body COM position set in xml-file                    (!! under world frame !!)
        self.LinPos[-1] = 0.23        # Position in z-axis
        self.LinVel = np.zeros((3, )) # Init main body COM linear velocity set in xml-file             (!! under world frame !!)
        self.LinVel_pre = self.LinVel # COM velocity in last timestep, used to approximate COM Position
        self.LinAcc = np.zeros((3, )) # Init main body COM linear acceleration set in xml-file         (!! under world frame !!)
        self.LinAcc[-1] += - 9.81
        self.LinAcc_pre = self.LinAcc # COM acceletation in last timestep, used to approximate COM linear velocity

        ## Related to Leg
        self.Ang = np.zeros((8, ))    # Leg Joint Actuator Length (turning angle)
        # self.Ang = self.sim.data.sensordata[6:14] # Leg Joint Actuator Length
        for i in range(4):
            self.Ang[i*2 + 0] = - self.sim.data.sensordata[6 + i*2 + 0] + 45 /180*math.pi
            self.Ang[i*2 + 1] = - self.sim.data.sensordata[6 + i*2 + 1] + 90 /180*math.pi
        self.Ang_pre = np.zeros((8, ))
        for i in range(len(self.Ang)):
            self.Ang_pre[i] = self.Ang[i]
        
        ## Related to Foot
        self.footPos = np.zeros((8,)) # Init foot position related to correspond hip joint             (!! under leg frame !!)
        self.footPos_pre = np.zeros((8,))
        # Leg Frame: x,z-coodinate is opposite to the x,z-coordinate of the body frame
        #            No y-coodinate right now -- 2 DOF Leg 
        for i in range(4):
            self.footPos[i*2: i*2+2] = np.array([1.38777878e-17, 1.97989899e-01])
            self.footPos_pre[i*2: i*2+2] = np.array([1.38777878e-17, 1.97989899e-01])

        self.footPos_3d = np.zeros((12, )) # Foot position under leg frame -> Foot Position under world frame
        for i in range(4):
            self.footPos_3d[i*3 + 0] = - self.footPos[i*2 + 0]
            self.footPos_3d[i*3 + 1] = 0
            self.footPos_3d[i*3 + 2] = - self.footPos[i*2 + 1]
            self.footPos_3d[i*3 : i*3+3] = self.R @ (self.footPos_3d[i*3 : i*3+3] + self.hipPos[:, i]) + self.LinPos
    
    def idle_motion(self):
        # To get a stable initial robot state
        # Using Virtuell Machine Control method (VMC)

        for loop in range(500):
            ctrlData = np.zeros((8, ))

            for i in range(4):

                l1 = self.l1
                l2 = self.l2

                ang1 = self.Ang[i*2 + 0]
                ang2 = self.Ang[i*2 + 1]

                # Current foot information
                footPos_Cur = self.footPos[i*2: i*2+2]
                footPos_Pre = self.footPos_pre[i*2: i*2+2]
                footPos_Vel = (footPos_Cur - footPos_Pre) / self.timestep

                # Desired foot postion related to the corresponding hip joint
                footPos_des = np.array([0, self.z0 + 0.010]) # Plus 0.010 of z-coordinate to offset the gravity influence 
                # print(footPos_des)

                # Use PD-control to control leg motion to follow the reference swing leg trajectory (Desired foot position)
                Kp = self.Kp_init
                Kd = self.Kd_init
                footPos_err = Kp * (footPos_des - footPos_Cur) + Kd * (0 - footPos_Vel)

                # Use Jacobi Matrix that gotten from kinematic analysis 
                # to compute the desired output torque of every motor (DOF) on the leg
                JT = np.array([[- l1*math.sin(ang1) - l2*math.sin(ang1+ang2), l1*math.cos(ang1) + l2*math.cos(ang1+ang2)], 
                                [- l1*math.sin(ang1+ang2), l2*math.cos(ang1+ang2)]])
                tau = JT @ footPos_err

                ctrlData[i*2: i*2+2] = - tau
            
            self.step(ctrlData)
            Xt = self.get_state()
        
        return Xt

    def step(self, ctrlData, printGRF=False):
        ## Transfer computed torque to Mujoco-Env and rendering
        self.sim.data.ctrl[:] = ctrlData
        self.sim.step()
        self.viewer.render()

        touch_GRF = np.around(self.sim.data.sensordata[23:27], decimals=4) # GRF [z]
        full_GRF = np.around(self.sim.data.efc_force[0:12], decimals=4) # GRF [z,y,x] * 4 Legs
        u_real = np.zeros_like(full_GRF)
        for i in range(len(touch_GRF)):
            if np.absolute(touch_GRF[i] - 0) > 1e-4:
                Fz_index = np.where(touch_GRF[i] == full_GRF)
                for ii in range(len(Fz_index[0])):
                    if Fz_index[0][ii] % 3 == 0:
                        u_real[3*i : 3*i + 3] = full_GRF[Fz_index[0][ii] : Fz_index[0][ii] + 3][::-1]
                        u_real[3*i] = - u_real[3*i]
                        u_real[3*i + 1] = - u_real[3*i + 1]
                        break
        if printGRF:
            print("touch sensor:", touch_GRF)
            print("Ground Reaction Force:", full_GRF)
            print("Real GRF:", u_real)

        return u_real
    
    def get_state(self):
        # Update the Robot variables #

        ## Related to Main Body COM
        # COM Euler Angle Position Measurement
        self.AngPos += (self.R @ self.AngVel + self.R @ self.AngVel_pre) * self.timestep / 2 # COM global euler angle is computed from COM local velocity
        self.R = expm(hatMap(self.AngPos))
        self.AngVel = self.sim.data.sensordata[0:3] # COM local velocity can be taken direktly from Gyro Sensor
        self.AngVel_pre = self.AngVel
        # print("COM Angle Velocity:", np.around(self.AngVel, decimals=4), "COM Angle Position:", np.around(self.AngPos, decimals=4))

        # COM Linear Position Measurement
        self.LinAcc = self.sim.data.sensordata[3:6]
        self.LinAcc = self.R @ self.LinAcc
        # if loop != 0:
        self.LinAcc[-1] += - 9.81

        self.LinVel += (self.LinAcc + self.LinAcc_pre) * self.timestep / 2
        self.LinAcc_pre = self.LinAcc
        self.LinPos += self.LinVel_pre * self.timestep + 1/2 * self.LinAcc * self.timestep**2
        self.LinVel_pre = self.LinVel

        Ref_LinPos = self.sim.data.sensordata[14:17]
        # print('------------------------------------------------------')
        # print("Real COM Linear Position:", np.around(Ref_LinPos, decimals=4))
        # print("COM Linear Position:", np.around(self.LinPos, decimals=4))
        # print('------')
        Ref_LinVel = self.sim.data.sensordata[17:20]
        # print("Real COM Linear Velocity:", np.around(Ref_LinVel, decimals=4))
        # print("COM Linear Velocity:", np.around(self.LinVel, decimals=4))
        # print('------')
        Ref_LinAcc = self.sim.data.sensordata[20:23]
        Ref_LinAcc[-1] += - 9.81
        # print("Real COM Linear Acceleration:", np.around(Ref_LinAcc, decimals=4))
        # print("COM Linear Acceleration:", np.around(self.LinAcc, decimals=4))

        self.LinPos = Ref_LinPos
        self.LinVel = Ref_LinVel

        ## Related to Leg
        l1 = self.l1
        l2 = self.l2
        for i in range(len(self.Ang)):
            self.Ang_pre[i] = self.Ang[i]
        for i in range(4): # Leg Joint Actuator Length
            self.Ang[i*2 + 0] = - self.sim.data.sensordata[6 + i*2 + 0] + 45 /180*math.pi
            self.Ang[i*2 + 1] = - self.sim.data.sensordata[6 + i*2 + 1] + 90 /180*math.pi 

        ## Related to Foot
        self.footPos_pre[:] = self.footPos[:]
        for i in range(4):
            # i = [0,1,2,3] correspond to [FL,FR,RL,RR]
            ang1 = self.Ang[i*2 + 0]
            ang2 = self.Ang[i*2 + 1]
            self.footPos[i*2: i*2+2] = np.array([l1*math.cos(ang1) + l2*math.cos(ang1+ang2), l1*math.sin(ang1) + l2*math.sin(ang1+ang2)])
        # print(self.footPos_pre, self.footPos)
        for i in range(4):
            self.footPos_3d[i*3 + 0] = - self.footPos[i*2 + 0]
            self.footPos_3d[i*3 + 1] = 0
            self.footPos_3d[i*3 + 2] = - self.footPos[i*2 + 1]
            self.footPos_3d[i*3 : i*3+3] = self.R @ (self.footPos_3d[i*3 : i*3+3] + self.hipPos[:, i]) + self.LinPos
        
        Xt = np.concatenate((self.LinPos, self.LinVel, self.R.flatten('F'), self.AngVel, self.footPos_3d))
            
        return Xt

    def computeTorque(self, FSM, Xt, Ut, Xd, printResult=False):
        # Force Control: Compute leg torques based on desired GRF and Jaccobi Matrix
        # Swing Leg Control: Compute leg torques based on reference swing leg trajectory (desired leg position)
        
        # Gait == 1 -> Bounding Gait:
        #   FSM - 1: Front Leg - Force Control; Rear Leg - Swing Leg Control
        #   FSM - 2: Front Leg - Swing Leg Control; Rear Leg - Swing Leg Control
        #   FSM - 3: Front Leg - Swing Leg Control; Rear Leg - Force Control
        #   FSM - 4: Front Leg - Swing Leg Control; Rear Leg - Swing Leg Control
        # Other Gaits:
        #   FSM - 1: Leg in stance phase -> Force Control
        #   FSM - 2: Leg in swing phase  -> Swing Leg Control

        if self.gait == 1:
            if FSM == 1:
                tau_FL = self.force_control(1, Xt, Ut)
                tau_FR = self.force_control(2, Xt, Ut)
                tau_RL = self.swing_leg_control(3, Xt, Xd, printResult)
                tau_RR = self.swing_leg_control(4, Xt, Xd, printResult)
            elif FSM == 2:
                tau_FL = self.swing_leg_control(1, Xt, Xd, printResult)
                tau_FR = self.swing_leg_control(2, Xt, Xd, printResult)
                tau_RL = self.swing_leg_control(3, Xt, Xd, printResult)
                tau_RR = self.swing_leg_control(4, Xt, Xd, printResult)
            elif FSM == 3:
                tau_FL = self.swing_leg_control(1, Xt, Xd, printResult)
                tau_FR = self.swing_leg_control(2, Xt, Xd, printResult)
                tau_RL = self.force_control(3, Xt, Ut)
                tau_RR = self.force_control(4, Xt, Ut)
            elif FSM == 4:
                tau_FL = self.swing_leg_control(1, Xt, Xd, printResult)
                tau_FR = self.swing_leg_control(2, Xt, Xd, printResult)
                tau_RL = self.swing_leg_control(3, Xt, Xd, printResult)
                tau_RR = self.swing_leg_control(4, Xt, Xd, printResult)

            CtrlData = np.concatenate((tau_FL, tau_FR, tau_RL, tau_RR))

        else:
            CtrlData = np.zeros((8, ))

            for i in range(4):
                if FSM[i] == 1: # stance phase
                    tau = self.force_control(i+1, Xt, Ut)
                else: # swing phase
                    tau = self.swing_leg_control(i+1, Xt, Xd, printResult)
                CtrlData[[i*2, i*2+1]] = tau

        return CtrlData

    def force_control(self, leg_ID, Xt, Ut):
        # leg_ID: 0, 1, 2, 3 for FL, FR, RL, RR
        # Ang: Turning angle of every actuator/motor (actuator length)
        #      (8, ): [FL_act1, FL_act2, FR_act1, ..., RR_act1, RR_act2]

        l1 = self.l1
        l2 = self.l2

        ang1 = self.Ang[(leg_ID - 1)*2 + 0]
        ang2 = self.Ang[(leg_ID - 1)*2 + 1]

        self.IsContact_Count[leg_ID - 1] = 0

        # Get desired GRFs (Ground reaction force) in 3-coordinate under body frame
        # For straight forward motion, there are only GRFs in x-, z- direction
        # -> Can be extended to motion with yaw-angle, but may need 3 DOFs on every leg
        GRF_w = np.array([Ut[(leg_ID - 1) * 3 + 0], 
                          Ut[(leg_ID - 1) * 3 + 1], 
                          Ut[(leg_ID - 1) * 3 + 2]])
        R = np.reshape(Xt[6:15], (3, 3), order='F')
        GRF_b = R.T @ GRF_w
        Fx = GRF_b[0] * 1
        Fz = GRF_b[2] * 1

        # Use Jacobi Matrix computed from kinematic analysis 
        # to compute the desired output torque of every motor (DOF) on the leg
        # based on the desired GRF computed by MPC 
        JT = np.array([[- l1*math.sin(ang1) - l2*math.sin(ang1+ang2), l1*math.cos(ang1) + l2*math.cos(ang1+ang2)], 
                        [- l1*math.sin(ang1+ang2), l2*math.cos(ang1+ang2)]])
        tau = JT @ np.array([Fx, Fz])

        return - tau

    def swing_leg_control(self, leg_ID, Xt, Xd, printResult):
        
        l1 = self.l1
        l2 = self.l2
        Kp = self.Kp
        Kd = self.Kd

        ang1 = self.Ang[(leg_ID - 1)*2 + 0]
        ang2 = self.Ang[(leg_ID - 1)*2 + 1]
        ang1_pre = self.Ang_pre[(leg_ID - 1)*2 + 0]
        ang2_pre = self.Ang_pre[(leg_ID - 1)*2 + 1]
        d_ang1 = (ang1 - ang1_pre) / self.timestep
        d_ang2 = (ang2 - ang2_pre) / self.timestep

        # Current foot information
        footPos_Cur = self.footPos[(leg_ID - 1)*2 : (leg_ID - 1)*2 + 2]
        footPos_Pre = self.footPos_pre[(leg_ID - 1)*2 : (leg_ID - 1)*2 + 2]
        footPos_Vel = (footPos_Cur - footPos_Pre) / self.timestep

        # Contact detection
        if self.sim.data.sensordata[23 + leg_ID - 1] > 1e-2:
            self.IsContact_Count[leg_ID - 1] += 1
        else:
            self.IsContact_Count[leg_ID - 1] = 0
        
        #   If the foot touches the ground too early, 
        #   keep the leg posture at the time of contact, 
        #   and no longer follow the swing leg trajectory in order to prevent the foot from bouncing，
        #   otherwise easily take the swing leg trajectory from MPC
        if self.IsContact_Count[leg_ID - 1] >= 1:
            if printResult:
                print("Contact detected!")
            Kp = 0.01 * Kp
            Kd = 0.01 * Kd

        # Desired foot postion related to the corresponding hip joint
        footPos_des_w = np.array([Xd[18+(leg_ID-1)*3, 0],
                                  Xd[18+(leg_ID-1)*3+1, 0],
                                  Xd[18+(leg_ID-1)*3+2, 0]])    # Foot Position under global frame
        COMPos_w = np.array(Xd[0:3, 0])                         # Desired COM position under global frame
        COM2foot_w = footPos_des_w - COMPos_w                   # Foot Position related to COM position under global frame

        R = np.reshape(Xd[6:15, 0], (3, 3), order='F')
        COM2foot_b = R.T @ COM2foot_w                           # Global Frame -> Body Frame

        COM2hip_b = self.hipPos[:, leg_ID- 1]                   # Hip Position related to COM position under body frame
        hip2foot_b = COM2foot_b - COM2hip_b
        hip2foot_l = np.array([-hip2foot_b[0], -hip2foot_b[2]]) # Body frame -> Leg frame
        footPos_des = hip2foot_l # Needed foot Position under leg frame

        # Use PD-control to control leg motion to follow the reference swing leg trajectory
        if printResult:
            print("Leg ID:", leg_ID, "FootPos_des:", footPos_des, "FootPos_cur:", footPos_Cur)

        c2 = (footPos_des[0]**2 + footPos_des[1]**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if c2 > 1:
            c2 = 1
        s2 = np.sqrt(1 - c2**2)
        theta2 = math.atan2(s2, c2)
        theta1 = math.atan2(footPos_des[1], footPos_des[0]) - math.atan2(l2 * s2, l1 + l2 * c2)

        theta2_err = Kp[1] * (theta2 - ang2) + Kd[1] * (0 - d_ang2)
        theta1_err = Kp[0] * (theta1 - ang1) + Kd[0] * (0 - d_ang1)
        tau = np.array([theta1_err, theta2_err])

        return - tau
    
    def circle_test(self):
        # Used for tuning of PD gain values
        l1 = self.l1
        l2 = self.l2
        Kp = self.Kp
        Kd = self.Kd
        ctrlData = np.zeros((8, ))
        # Circle target trajectory
        theta = np.linspace(90, 450, num=100)
        theta = theta /180*math.pi
        a = 0.05
        b = 0.02
        footPos_desTraj = np.zeros((len(theta), 2))
        center = np.array([0, 0.178])
        for i in range(len(theta)):
            x = a * math.cos(theta[i])
            y = b * math.sin(theta[i])
            cur_point  = center + np.array([x, y])
            footPos_desTraj[i, 0:2] = cur_point

        for loop in range(100):
            print("--------------Iteration:", loop, "--------------")
            footPos_des = footPos_desTraj[loop, :]
            for i in range(4):
                # i = [0,1,2,3] correspond to [FL,FR,RL,RR]
                ang1 = self.Ang[i*2 + 0]
                ang2 = self.Ang[i*2 + 1]
                ang1_pre = self.Ang_pre[i*2 + 0]
                ang2_pre = self.Ang_pre[i*2 + 1]
                d_ang1 = (ang1 - ang1_pre) / self.timestep
                d_ang2 = (ang2 - ang2_pre) / self.timestep

                c2 = (footPos_des[0]**2 + footPos_des[1]**2 - l1**2 - l2**2) / (2 * l1 * l2)
                s2 = np.sqrt(1 - c2**2)
                theta2 = math.atan2(s2, c2)
                theta1 = math.atan2(footPos_des[1], footPos_des[0]) - math.atan2(l2 * s2, l1 + l2 * c2)
                theta2_err = Kp[1] * (theta2 - ang2) + Kd[1] * (0 - d_ang2)
                theta1_err = Kp[0] * (theta1 - ang1) + Kd[0] * (0 - d_ang1)
                tau = np.array([theta1_err, theta2_err])
                ctrlData[i*2+0:i*2+2] = - tau
                if i == 0:
                    print("Desired Angle:", theta1, theta2, "Real Angle:", ang1, ang2)
                    print("Previous Angle:", ang1_pre, ang2_pre)

            self.step(ctrlData)
            self.get_state()