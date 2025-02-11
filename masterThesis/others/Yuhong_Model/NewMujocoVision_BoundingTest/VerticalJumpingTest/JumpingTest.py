from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math

from LegModel.legs import LegModel

# init part
RUN_STEPS = 15 # 100 for Video, 10 for plot

model = load_model_from_path("../models/dynamic_biped_quad_val.xml") 
# dynamic_4l.xml || dynamic_4l_VerticalSpine.xml
sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()
sim.set_state(sim_state)

ctrlData = [0, 0, 0, 0, 0, 0, 0, 0]
# ctrlData = [0, 0, 0, 0]
# ctrlData = [-0.46, -0.84, -0.46, -0.84, -0.66, -0.76, -0.66, -0.76, 0, 0, 0, 0]

leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
fl_left = LegModel(leg_params)
fl_right = LegModel(leg_params)
hl_left = LegModel(leg_params)
hl_right = LegModel(leg_params)

# init part for plot
legPosName = [["leg_link_fl", "ankle_fl"],
			["leg_link_fr", "ankle_fr"],
			["leg_link_rl", "ankle_rl"],
			["leg_link_rr", "ankle_rr"]]
legGivenPoint_x_fp = []
legGivenPoint_x_hp = []
legGivenPoint_y_fp = []
legGivenPoint_y_hp = []
legRealPoint_x = [[],[],[],[]]
legRealPoint_y = [[],[],[],[]]

# idle motion
for i in range(500):  
    sim.data.ctrl[:] = ctrlData
    sim.step()
    viewer.render()

# ctrldata computation
def computer_ctrldata(fp, hp):
    foreLeg_left_q = fl_left.pos_2_angle(0.0139, fp)
    foreLeg_right_q = fl_right.pos_2_angle(0.0139, fp)
    # foreLeg_left_q = [0, 0]
    # foreLeg_right_q = [0, 0]
    # hindLeg_left_q = hl_left.pos_2_angle(-0.000, hp)
    # hindLeg_right_q = hl_right.pos_2_angle(-0.000, hp)
    hindLeg_left_q = [0, 0]
    hindLeg_right_q = [0, 0]
    ctrlData = []
    ctrlData.extend(foreLeg_left_q)
    ctrlData.extend(foreLeg_right_q)
    ctrlData.extend(hindLeg_left_q)
    ctrlData.extend(hindLeg_right_q)
    for i in range(5):
        ctrlData.append(0)
    
    return ctrlData

# init motion
init_fp = -0.035 # -0.035
init_hp = -0.044 # -0.044
# ctrlData = computer_ctrldata(init_fp, init_hp)
# # print(ctrlData)
# for i in range(1000): 
#     sim.data.ctrl[:] = ctrlData
#     # ctrlData = [-0.84 -0.46 -0.84 -0.46 -0.76 -0.66 -0.76 -0.66]
#     sim.step()
#     viewer.render()
#     # if i % 100 == 0:
#     #     print(sim.data.sensordata[0:8])

# run motion
for i in range(RUN_STEPS):
    range_run = 600
    takeoff_step = range_run / 12 # 6
    print("Cycle-----------------------------------", i)
    for j in range(range_run):

        if j < (takeoff_step):
            fp = - 0.050
            hp = - 0.059
            ctrlData = computer_ctrldata(fp, hp)
            ctrlData = ctrlData[0:4]
            ctrlData = np.hstack((ctrlData, ctrlData))
            # fp = (0.035-0.050) / takeoff_step * j - 0.035 # 0.035-0.050 -0.035
            # hp = (0.044-0.059) / takeoff_step * j - 0.044 # 0.044-0.059 -0.044
            # ctrlData = computer_ctrldata(fp, hp)   
        else: 
            fp = init_fp
            hp = init_hp
            ctrlData = computer_ctrldata(fp, hp)
            ctrlData = ctrlData[0:4]
            ctrlData = np.hstack((ctrlData, ctrlData))
    
        sim.data.ctrl[:] = ctrlData
        sim.step()
        viewer.render()

        if j < (takeoff_step):
            print("Contact:", sim.data.ncon, "Torque:", np.around(sim.data.actuator_force[0:2], decimals=5), "Force_r:", np.around(sim.data.efc_force[6:18], 3))

        # legGivenPoint_x_fp.append(fp)
        # legGivenPoint_x_hp.append(hp)

        # for i in range(4):
        #     originPoint = sim.data.get_site_xpos(legPosName[i][0])
        #     currentPoint = sim.data.get_site_xpos(legPosName[i][1])
        #     tX = currentPoint[1]-originPoint[1]
        #     tY = currentPoint[2]-originPoint[2]
        #     legRealPoint_x[i].append(tX)
        #     legRealPoint_y[i].append(tY)

# Plot
# if RUN_STEPS == 10:
#     legGivenPoint_x_fp = np.array(legGivenPoint_x_fp)
#     legGivenPoint_x_hp = np.array(legGivenPoint_x_hp)
#     legGivenPoint_y_fp = np.zeros_like(legGivenPoint_x_fp)
#     legGivenPoint_y_hp = np.zeros_like(legGivenPoint_x_hp)
#     legRealPoint_x = np.array(legRealPoint_x)
#     legRealPoint_y = np.array(legRealPoint_y)

#     plt.plot(legGivenPoint_y_fp, legGivenPoint_x_fp)
#     plt.plot(legRealPoint_x[0], legRealPoint_y[0])
#     plt.grid()
#     plt.title("Front Left Leg")
#     plt.show()

#     plt.plot(legGivenPoint_y_hp, legGivenPoint_x_hp)
#     plt.plot(legRealPoint_x[2], legRealPoint_y[2])
#     plt.grid()
#     plt.title("Rear Left Leg")
#     plt.show()