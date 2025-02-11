# Load xml-model to mujoco
from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas
from LegModel.legs import LegModel

model = load_model_from_path("../models/dynamic_biped_quad_val.xml") # test.xml / dynamic_biped_val.xml / dynamic_biped_quad_val.xml
sim = MjSim(model)
viewer = MjViewer(sim)
#'''
viewer.cam.azimuth = 0
viewer.cam.lookat[0] += 0.25
viewer.cam.lookat[1] += -0.5
viewer.cam.distance = model.stat.extent * 0.5
viewer.run_speed = 0.02
#'''
sim_state = sim.get_state()
sim.set_state(sim_state)

# ctrlData = [0, 0, 0, 0]
ctrlData = [0, 0, 0, 0, 0, 0, 0, 0]
siteName = []
for i in range(sim.model.nsite):
    siteName.append(sim.model.site_id2name(i))
    if siteName[-1] == None:
        siteName[-1] = 'None'
site_KeyIndex = np.array([siteName.index("ankle_fl"), siteName.index("ankle_fr"),
                            siteName.index("leg_link_fl"), siteName.index("leg_link_fr")])

leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
legModel = LegModel(leg_params)

# Init Simulation
for i in range(500):
    sim.data.ctrl[:] = ctrlData
    sim.step()
    viewer.render()
    # print(np.around(sim.data.efc_force[6:9], 3))

# Start Simulation (Vertical Jumping Test)
def bezier(time, alpha=1, peak=0.5):
    if time <= peak:
        p = [0.0, 0.8, 1.0, 1.0]
        b = p[0]*(1-time)**3 + 3*p[1]*time*(1-time)**2 + 3*p[2]*time**2*(1-time) + p[3]*time**3
        b *= alpha
    else:
        p = [1.0, 1.0, 0.8, 0.0]
        b = p[0]*(1-time)**3 + 3*p[1]*time*(1-time)**2 + 3*p[2]*time**2*(1-time) + p[3]*time**3
        b *= alpha

    return b

def footendForce_2_torque(force_d, q1, q2):
    j11 = - (31*math.sin(q1))/300 - (7*math.cos(math.acos((25*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2))) + math.asin(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)))*(((25*((31*math.cos(q1)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/500 + (31*math.sin(q1)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/500))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (25*((31*math.cos(q1)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/500 + (31*math.sin(q1)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/500)*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(4*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - (625*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000)**2)/(4*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + 4*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2) - ((31*math.sin(q1))/(1000*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (((31*math.cos(q1)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/500 + (31*math.sin(q1)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/500)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - ((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2)))/75
    j12 = (413*math.cos(q2))/15000 + (7*math.cos(math.acos((25*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2))) + math.asin(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)))*(((25*((59*math.cos(q2)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/2500 - (59*math.sin(q2)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/2500))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (25*((59*math.cos(q2)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/2500 - (59*math.sin(q2)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/2500)*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(4*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - (625*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000)**2)/(4*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + 4*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2) - ((59*math.cos(q2))/(5000*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (((59*math.cos(q2)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/2500 - (59*math.sin(q2)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/2500)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - ((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2)))/75
    j21 = (31*math.cos(q1))/300 - (7*math.sin(math.acos((25*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2))) + math.asin(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)))*(((25*((31*math.cos(q1)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/500 + (31*math.sin(q1)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/500))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (25*((31*math.cos(q1)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/500 + (31*math.sin(q1)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/500)*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(4*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - (625*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000)**2)/(4*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + 4*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2) - ((31*math.sin(q1))/(1000*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (((31*math.cos(q1)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/500 + (31*math.sin(q1)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/500)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - ((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2)))/75
    j22 = (413*math.sin(q2))/15000 + (7*math.sin(math.acos((25*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2))) + math.asin(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)))*(((25*((59*math.cos(q2)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/2500 - (59*math.sin(q2)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/2500))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (25*((59*math.cos(q2)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/2500 - (59*math.sin(q2)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/2500)*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000))/(4*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - (625*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2 + 11/8000)**2)/(4*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + 4*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2) - ((59*math.cos(q2))/(5000*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(1/2)) - (((59*math.cos(q2)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/2500 - (59*math.sin(q2)*((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625))/2500)*((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000))/(2*(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2)**(3/2)))/(1 - ((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2/(((31*math.cos(q1))/1000 + (59*math.sin(q2))/5000)**2 + ((59*math.cos(q2))/5000 - (31*math.sin(q1))/1000 + 8/625)**2))**(1/2)))/75
    j = np.array([[j11, j12], [j21, j22]])
    j_T = np.transpose(j)
    torque = np.dot(j_T, force_d)

    return torque

init_height = sim.data.subtree_com[0,:][-1]
count_time = 0
discretNum = 30
periodTime = 0
wholePeriod = 150
direction = True
save_data = False

footend_pos_list = np.empty((0,3))
contact_detected_list = np.empty((0))
force_desired_list = np.empty((0,2))
force_detected_list = np.empty((0,3))
# while True:
for i in range(wholePeriod):
    periodTime += 1
    if periodTime > wholePeriod:
        periodTime = 1

    print(periodTime)
    if periodTime <= 2*discretNum:
        count_time += 1/discretNum
        # print(count_time)
        if count_time > 1:
            count_time = 1/discretNum
            direction = not direction

        Fy = bezier(count_time, alpha=0, peak=0.5)
        Fz = bezier(count_time, alpha=5, peak=0.5)

        q1 = sim.data.actuator_length[0]
        q2 = sim.data.actuator_length[1]
        torque_d = - footendForce_2_torque(np.array([[Fy], [Fz]]), q1, q2) # *4
        if not direction:
            torque_d *= -1
            Fz *= -1
            Fy *= -1

        ctrlData = np.append(np.transpose(torque_d), np.transpose(torque_d), 1)[0]
        # print(ctrlData)
        ctrlData = np.hstack((ctrlData, ctrlData))
    else:
        ctrlData = [0, 0, 0, 0, 0, 0, 0, 0]
        # ctrlData = [0, 0, 0, 0]

    sim.data.ctrl[:] = ctrlData
    sim.step()
    viewer.render()

    site_pos = sim.data.site_xpos
    footendPos_fl = site_pos[site_KeyIndex[0]]
    footendPos_fr = site_pos[site_KeyIndex[1]]

    footend_pos_list = np.append(footend_pos_list, np.array([np.around(footendPos_fl[0:3],decimals=5)]), 0)
    contact_detected_list = np.append(contact_detected_list, np.array([sim.data.ncon]), 0)
    force_desired_list = np.append(force_desired_list, np.array([np.around(np.array([Fz, Fy]), decimals=3)]), 0)
    force_detected_list = np.append(force_detected_list, np.array([np.around(sim.data.efc_force[6:9], 3)]), 0)
    # if i%10 == 0:
    # print("Footend Pos:", np.around(footendPos_fl[0:3],decimals=5))
    if periodTime > 2*discretNum:
        dir = "Empty"
    else:
        if direction:
            dir = "Up"
        else:
            dir = "Down"
    
    cur_height = sim.data.subtree_com[0,:][-1]
    dif_height = np.around((cur_height-init_height), decimals=5)
    print("Direction:", dir, "COM Height:", dif_height, "Contact:", sim.data.ncon, "Torque:", np.around(ctrlData[0:2], decimals=5), "Force_d:", np.around(np.array([Fz, Fy]), decimals=3), "Force_r:", np.around(sim.data.efc_force[6:9], 3))

if save_data:
    df = pandas.DataFrame({"Footend Pos":footend_pos_list.tolist(), "Contact detected":contact_detected_list.tolist(), "Desired Force":force_desired_list.tolist(), "Detected Force":force_detected_list.tolist()})
    df.to_excel("JacobianMatricVerification.xlsx")