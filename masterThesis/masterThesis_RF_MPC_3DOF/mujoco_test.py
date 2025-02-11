# Load xml-model to mujoco
from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import math
import pandas
from fcns.quaternion_to_euler import quaternion_to_euler
from scipy.linalg import expm
from fcns.hatMap import hatMap

model = load_model_from_path("models/model_origin.xml")
# Sensordata: 
# [0:2] - COM Angle Velocity [x,y,z]
# [3:5] - COM Linear Acceleration [x,y,z]
# [6:13] - [FL,FR,RL,RR]*[UpMotor,DownMotor]
sim = MjSim(model)
viewer = MjViewer(sim)

viewer.cam.azimuth = 270
viewer.cam.elevation = -1
viewer.cam.lookat[0] += -0
viewer.cam.lookat[1] += -0
viewer.cam.lookat[2] += 0
viewer.cam.distance = model.stat.extent * 0.5
viewer._run_speed = 0.05

sim_state = sim.get_state()
sim.set_state(sim_state)

l1 = 0.14
l2 = 0.14
m1 = 0.055
m2 = 0.055
g = 9.81
Kp = np.array([1600, 1600]) # K_p_fb
Kd = np.array([5, 5]) # K_d_fb
Kp_ff = np.array([2, 2])
Kd_ff = np.array([0.2, 0.2])
timestep = 0.001

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

# Origin trajectory
''' 
footPos_des_start = np.array([0.00, 0.20]) # desired foot position (x, z)
footPos_des_end = np.array([-0.10, 0.10]) # desired foot position (x, z)
footPos_desTraj = np.linspace(footPos_des_start, footPos_des_end, num=10)
'''

footPos_des = footPos_desTraj[0, :]
footPos_des_pre = footPos_des
footVel_des = (footPos_des - footPos_des_pre) / timestep
footVel_des_pre = footVel_des
footAcc_des = (footVel_des - footVel_des_pre) / timestep

footPos_pre = np.zeros((8,))
for i in range(4):
    footPos_pre[i*2: i*2+2] = np.array([1.38777878e-17, 1.97989899e-01]) # origin foot position

ctrlData = np.zeros((8, ))

# Parameters related to body COM
AngPos = np.zeros((3, ))
R = expm(hatMap(AngPos))
AngVel = np.zeros((3, ))
AngVel_pre = AngVel
LinPos = np.zeros((3, ))
LinPos[-1] = 0.23
LinVel = np.zeros((3, ))
LinVel_pre = LinVel
LinAcc = np.zeros((3, ))
LinAcc_pre = LinAcc

footPos_cur_list_x = []
footPos_cur_list_z = []
footPos_vel_list_x = []
footPos_vel_list_z = []
footPos_des_list_x = []
footPos_des_list_z = []
used_torque_list_1 = []
used_torque_list_2 = []

Ang = sim.data.sensordata[6:14]
Ang_pre = np.zeros_like(Ang)
# change to the init foot position
for i in range(4):
    Ang_pre[i*2 + 0] = - Ang[i*2 + 0] + 45 /180*math.pi
    Ang_pre[i*2 + 1] = - Ang[i*2 + 1] + 90 /180*math.pi

theta_pre = np.zeros_like(Ang_pre)
for i in range(len(Ang_pre)):
    theta_pre[i] = Ang_pre[i]

for loop in range(100):

    print("--------------Iteration:", loop, "--------------")

    ## Test for COM Angle Position Measurement
    AngPos += (R @ AngVel + R @ AngVel_pre) * timestep / 2 # COM global euler angle is computed from COM local velocity
    R = expm(hatMap(AngPos))
    AngVel = sim.data.sensordata[0:3] # COM local velocity can be taken direktly from Gyro Sensor
    AngVel_pre = AngVel
    # print("COM Angle Velocity:", np.around(AngVel, decimals=4), "COM Angle Position:", np.around(AngPos, decimals=4))

    ## Compare with the true COM Angle Position
    # Quat = sim.data.sensordata[17:21]
    # AngPos_w = quaternion_to_euler(Quat)
    # print("COM Angle Position in world frame:", np.around(AngPos_w, decimals=4))

    ## Test for COM Linear Position Measurement
    LinAcc = sim.data.sensordata[3:6]
    LinAcc = R @ LinAcc
    if loop != 0:
        LinAcc[-1] += - 9.81

    LinVel += (LinAcc + LinAcc_pre) * timestep / 2
    LinAcc_pre = LinAcc
    LinPos += LinVel_pre * timestep + 1/2 * LinAcc * timestep**2
    LinVel_pre = LinVel
    
    ## Compare with the true COM Linear Position
    Ref_LinPos = sim.data.sensordata[14:17]
    # print("Real COM Linear Position:", np.around(Ref_LinPos, decimals=4))
    # print("COM Linear Position:", np.around(LinPos, decimals=4))
    # print('------')
    Ref_LinVel = sim.data.sensordata[17:20]
    # print("Real COM Linear Velocity:", np.around(Ref_LinVel, decimals=4))
    # print("COM Linear Velocity:", np.around(LinVel, decimals=4))
    # print('------')
    Ref_LinAcc = sim.data.sensordata[20:23]
    # Ref_LinAcc[-1] += - 9.81
    # print("Real COM Linear Acceleration:", np.around(Ref_LinAcc, decimals=4))
    # print("COM Linear Acceleration:", np.around(LinAcc, decimals=4))

    ## Reference swing leg trajectory
    footPos_des = footPos_desTraj[loop, :]
    footVel_des = (footPos_des - footPos_des_pre) / timestep
    footAcc_des = (footVel_des - footVel_des_pre) / timestep
    footPos_des_pre = footPos_des
    footVel_des_pre = footVel_des
    # print("Ref_Pos:", footPos_des, "Ref_Vel:", footVel_des, "Ref_Acc:", footAcc_des)

    ## Body Inertia
    # print(sim.model.body_inertia) # (body in the order defined in XML, [x, y, z])

    ## Test of Foot Acceleration
    footAcc_global = sim.data.sensordata[23:26]
    # print("          Real Foot Accelaration:", np.around(footAcc_global, decimals=4))

    ## PD Control for Swing Leg Control (VMC + Leg Dynamic)
    Ang = sim.data.sensordata[6:14] # current foot joint actuator length
    print("Ang:", Ang)
    for i in range(4):
        # i = [0,1,2,3] correspond to [FL,FR,RL,RR]

        ang1 = - Ang[i*2 + 0] + 45 /180*math.pi
        ang2 = - Ang[i*2 + 1] + 90 /180*math.pi
        d_ang1 = (ang1 - Ang_pre[i*2 + 0]) / timestep
        d_ang2 = (ang2 - Ang_pre[i*2 + 1]) / timestep
        if i == 0:
            print(ang1, ang2, Ang_pre[i*2 + 0], Ang_pre[i*2 + 1])
        Ang_pre[i*2 + 0] = ang1
        Ang_pre[i*2 + 1] = ang2
        

        footPos_cur = np.array([l1*math.cos(ang1) + l2*math.cos(ang1+ang2), l1*math.sin(ang1) + l2*math.sin(ang1+ang2)])
        footPos_vel = (footPos_cur - footPos_pre[i*2: i*2+2]) / timestep
        footPos_pre[i*2: i*2+2] = footPos_cur

        footPos_err = Kp * (footPos_des - footPos_cur) + Kd * (0 - footPos_vel)

        # if i == 0:
        #     print("Leg ID:", i, "Current Foot Position:", np.around(footPos_cur, decimals=4), "Desired Foot Position:", footPos_des)
        #     print("          Current Foot Velocity:", np.around(footPos_vel, decimals=4), "Desired Foot Velocity:", footVel_des)
        #     print("          Desired Foot Acceleration:", np.around(footAcc_des, decimals=4))

        JT = np.array([[- l1*math.sin(ang1) - l2*math.sin(ang1+ang2), l1*math.cos(ang1) + l2*math.cos(ang1+ang2)], 
                        [- l1*math.sin(ang1+ang2), l2*math.cos(ang1+ang2)]])
        
        # VMC Term
        # 1: Use Jacobi matrix
        # tau = JT @ footPos_err

        # 2: Use Inverse Kinematic
        c2 = (footPos_des[0]**2 + footPos_des[1]**2 - l1**2 - l2**2) / (2 * l1 * l2)
        s2 = np.sqrt(1 - c2**2)
        theta2 = math.atan2(s2, c2)
        theta1 = math.atan2(footPos_des[1], footPos_des[0]) - math.atan2(l2 * s2, l1 + l2 * c2)
        # theta2_pre = theta_pre[i*2 + 1]
        # theta1_pre = theta_pre[i*2 + 0]
        # theta2_vel = (theta2 - theta2_pre) / timestep
        # theta1_vel = (theta1 - theta1_pre) / timestep
        # theta_pre[i*2:i*2+2] = np.array([theta1, theta2])
        theta2_err = Kp[1] * (theta2 - ang2) + Kd[1] * (0 - d_ang2)
        theta1_err = Kp[0] * (theta1 - ang1) + Kd[0] * (0 - d_ang1)
        tau = np.array([theta1_err, theta2_err])
        # tau = - np.array([-(theta1-45/180*math.pi), -(theta2-90/180*math.pi)])

        # For directly GRF force control
        tau = JT @ np.array([0, 15])

        # Leg Dynamic Term
        M = np.array([[(l1**2*m1)/4 + l1**2*m2 + (l2**2*m2)/4 + l1*l2*m2*math.cos(ang2) + 2, (m2*l2**2)/4 + (l1*m2*math.cos(ang2)*l2)/2 + 1],
                      [(m2*l2**2)/4 + (l1*m2*math.cos(ang2)*l2)/2 + 1, (m2*l2**2)/4 + 1]])
        V = np.array([-(l1*l2*m2*math.sin(ang2)*(2*d_ang1 + d_ang2)*d_ang2)/2, (l1*l2*m2*math.sin(ang2)*d_ang1**2)/2])
        G = np.array([- g*m2*((l2*math.cos(ang1 + ang2))/2 + l1*math.cos(ang1)) - (l1*g*m1*math.cos(ang1))/2, -(l2*g*m2*math.cos(ang1 + ang2))/2])
        J = np.array([[- l1*math.sin(ang1) - l2*math.sin(ang1 + ang2), -l2*math.sin(ang1 + ang2)],
                      [  l2*math.cos(ang1 + ang2) + l1*math.cos(ang1),  l2*math.cos(ang1 + ang2)]])
        dJ = np.array([[- l2*math.cos(ang1 + ang2)*(d_ang1 + d_ang2) - l1*math.cos(ang1)*d_ang1, -l2*math.cos(ang1 + ang2)*(d_ang1 + d_ang2)], 
                       [- l2*math.sin(ang1 + ang2)*(d_ang1 + d_ang2) - l1*math.sin(ang1)*d_ang1, -l2*math.sin(ang1 + ang2)*(d_ang1 + d_ang2)]])
        dq = np.array([d_ang1, d_ang2])

        # a_ref = footAcc_des + Kp_ff * (footPos_des - footPos_cur) + Kd_ff * (footVel_des - footPos_vel)
        a_ref = np.array([5, 5])
        tau_dy = M @ J**(-1) @ (a_ref - dJ @ dq) + V + G
        # print(M, J**(-1), dJ@dq, V, G)
        if i == 0:
            print("          VMC Torque:", tau, "Dyn Torque:", tau_dy)
        tau_dy[tau_dy > 15] = 15
        tau_dy[tau_dy < -15] = -15

        ctrlData[i*2+0:i*2+2] = - tau

    print("Computed Leg Torque:", np.around(ctrlData, decimals=4))

    footPos_cur_list_x.append(footPos_cur[0])
    footPos_cur_list_z.append(footPos_cur[1])
    footPos_des_list_x.append(footPos_des[0])
    footPos_des_list_z.append(footPos_des[1])
    footPos_vel_list_x.append(footPos_vel[0])
    footPos_vel_list_z.append(footPos_vel[1])
    used_torque_list_1.append(ctrlData[0])
    used_torque_list_2.append(ctrlData[1])

    ## Transfer computed torque to Mujoco-Env and rendering
    sim.data.ctrl[:] = ctrlData
    sim.step()
    viewer.render()

    ## GRF daten
    # print("Ground Reaction Force:", np.around(sim.data.efc_force[[2, 0]], decimals=4))
    print("Real Output Torque:", sim.data.actuator_force)

#####################################################################
## Static Plot
fig, axs = plt.subplots(3, 2)
supTitle = 'PD Tuning Test with Kp: ' + np.array2string(Kp) + ' and Kd: ' + np.array2string(Kd)
fig.suptitle(supTitle)

axs[0,0].plot(range(loop+1), footPos_cur_list_x)
axs[0,0].plot(range(loop+1), footPos_des_list_x)
axs[0,0].set_title("Foot Position x")
axs[0,0].grid()
axs[0,0].set_xlabel('Time (s)')

axs[0,1].plot(range(loop+1), footPos_cur_list_z)
axs[0,1].plot(range(loop+1), footPos_des_list_z)
axs[0,1].set_title("Foot Position z")
axs[0,1].grid()
axs[0,1].set_xlabel('Time (s)')

axs[1,0].plot(range(loop+1), footPos_vel_list_x)
axs[1,0].plot(range(loop+1), footPos_vel_list_x)
axs[1,0].set_title("Foot Velocity x")
axs[1,0].grid()
axs[1,0].set_xlabel('Time (s)')

axs[1,1].plot(range(loop+1), footPos_vel_list_x)
axs[1,1].plot(range(loop+1), footPos_vel_list_x)
axs[1,1].set_title("Foot Velocity z")
axs[1,1].grid()
axs[1,1].set_xlabel('Time (s)')

axs[2,0].plot(range(loop+1), used_torque_list_1)
axs[2,0].plot(range(loop+1), used_torque_list_2)
axs[2,0].set_title("Output Torque")
axs[2,0].grid()
axs[2,0].set_xlabel('Time (s)')

plt.show()

#####################################################################
## Animation
fig, ax = plt.subplots()
line1, = ax.plot(footPos_cur_list_x, footPos_cur_list_z)
line2, = ax.plot(footPos_des_list_x, footPos_des_list_z)
def animate(i):
    line1.set_xdata(footPos_cur_list_x[0:i+1])
    line1.set_ydata(footPos_cur_list_z[0:i+1])
    line2.set_xdata(footPos_des_list_x[0:i+1])
    line2.set_ydata(footPos_des_list_z[0:i+1])
    return line1, line2
ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=100)

ani.save("simple_animation.gif", dpi=300,
         writer=PillowWriter(fps=5))
plt.show()