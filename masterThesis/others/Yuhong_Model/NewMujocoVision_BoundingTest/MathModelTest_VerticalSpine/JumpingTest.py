from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math

from LegModel.legs import LegModel

# init part
RUN_STEPS = 100 # 100 for Video, 10 for plot

model = load_model_from_path("../models/dynamic_4l_VerticalSpine.xml") 
# dynamic_4l.xml || dynamic_4l_VerticalSpine.xml
sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()
sim.set_state(sim_state)

ctrlData = [0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, 0,0,0,0, -0.3]
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
spineStartSite = "body_ss"
spineEndSite = "body_ss2"
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

    startPoint = sim.data.get_site_xpos(spineStartSite)
    endPoint = sim.data.get_site_xpos(spineEndSite)
    delta_y = endPoint[1] - startPoint[1]
    delta_z = - (endPoint[2] - startPoint[2])
    theta_s = math.atan(delta_z/delta_y) / math.pi * 180
    # print("startPoint:", np.round(startPoint, 3))
    # print("endpoint:", np.round(endPoint, 3))
    #print("dy:", np.round(delta_y, 3), " | dz:", np.round(delta_z, 3), " | ds:", np.round(theta_s, 3))

    legShouPoint_fl = sim.data.get_site_xpos(legPosName[0][0])
    diff_fl = startPoint - legShouPoint_fl
    legShouPoint_fr = sim.data.get_site_xpos(legPosName[1][0])
    diff_fr = startPoint - legShouPoint_fr
    legShouPoint_rl = sim.data.get_site_xpos(legPosName[2][0])
    diff_rl = endPoint - legShouPoint_rl
    legShouPoint_rr = sim.data.get_site_xpos(legPosName[3][0])
    diff_rr = endPoint - legShouPoint_rr

    print('-------------------------------------')
    print("relativ position of fl-leg: ", "x: ", np.round(diff_fl[0], 3), "y: ", np.round(diff_fl[1], 3), "z: ", np.round(diff_fl[1], 3))
    print("relativ position of fr-leg: ", "x: ", np.round(diff_fr[0], 3), "y: ", np.round(diff_fr[1], 3), "z: ", np.round(diff_fr[1], 3))
    print("relativ position of rl-leg: ", "x: ", np.round(diff_rl[0], 3), "y: ", np.round(diff_rl[1], 3), "z: ", np.round(diff_rl[1], 3))
    print("relativ position of rr-leg: ", "x: ", np.round(diff_rr[0], 3), "y: ", np.round(diff_rr[1], 3), "z: ", np.round(diff_rr[1], 3))

# # ctrldata computation
# def computer_ctrldata(fp, hp):
#     foreLeg_left_q = fl_left.pos_2_angle(-0.000, fp)
#     foreLeg_right_q = fl_right.pos_2_angle(-0.000, fp)
#     # foreLeg_left_q = [0, 0]
#     # foreLeg_right_q = [0, 0]
#     # hindLeg_left_q = hl_left.pos_2_angle(-0.000, hp)
#     # hindLeg_right_q = hl_right.pos_2_angle(-0.000, hp)
#     hindLeg_left_q = [0, 0]
#     hindLeg_right_q = [0, 0]
#     ctrlData = []
#     ctrlData.extend(foreLeg_left_q)
#     ctrlData.extend(foreLeg_right_q)
#     ctrlData.extend(hindLeg_left_q)
#     ctrlData.extend(hindLeg_right_q)
#     for i in range(5):
#         ctrlData.append(0)
    
#     return ctrlData

# # init motion
# init_fp = -0.035 # -0.035
# init_hp = -0.044 # -0.044
# ctrlData = computer_ctrldata(init_fp, init_hp)
# # print(ctrlData)
# for i in range(1000): 
#     sim.data.ctrl[:] = ctrlData
#     # ctrlData = [-0.84 -0.46 -0.84 -0.46 -0.76 -0.66 -0.76 -0.66]
#     sim.step()
#     viewer.render()
#     # if i % 100 == 0:
#     #     print(sim.data.sensordata[0:8])

# # run motion
# for i in range(RUN_STEPS):
#     range_run = 300
#     takeoff_step = range_run / 6 # 6
#     for j in range(range_run):

#         if j < (takeoff_step):
#             fp = (0.035-0.050) / takeoff_step * j - 0.035 # 0.035-0.050 -0.035
#             hp = (0.044-0.059) / takeoff_step * j - 0.044 # 0.044-0.059 -0.044
#             ctrlData = computer_ctrldata(fp, hp)   
#         else: 
#             fp = init_fp
#             hp = init_hp
#             ctrlData = computer_ctrldata(fp, hp)
    
#         sim.data.ctrl[:] = ctrlData
#         sim.step()
#         viewer.render()

#         legGivenPoint_x_fp.append(fp)
#         legGivenPoint_x_hp.append(hp)

#         for i in range(4):
#             originPoint = sim.data.get_site_xpos(legPosName[i][0])
#             currentPoint = sim.data.get_site_xpos(legPosName[i][1])
#             tX = currentPoint[1]-originPoint[1]
#             tY = currentPoint[2]-originPoint[2]
#             legRealPoint_x[i].append(tX)
#             legRealPoint_y[i].append(tY)

# Plot
if RUN_STEPS == 10:
    legGivenPoint_x_fp = np.array(legGivenPoint_x_fp)
    legGivenPoint_x_hp = np.array(legGivenPoint_x_hp)
    legGivenPoint_y_fp = np.zeros_like(legGivenPoint_x_fp)
    legGivenPoint_y_hp = np.zeros_like(legGivenPoint_x_hp)
    legRealPoint_x = np.array(legRealPoint_x)
    legRealPoint_y = np.array(legRealPoint_y)

    plt.plot(legGivenPoint_y_fp, legGivenPoint_x_fp)
    plt.plot(legRealPoint_x[0], legRealPoint_y[0])
    plt.grid()
    plt.title("Front Left Leg")
    plt.show()

    plt.plot(legGivenPoint_y_hp, legGivenPoint_x_hp)
    plt.plot(legRealPoint_x[2], legRealPoint_y[2])
    plt.grid()
    plt.title("Rear Left Leg")
    plt.show()