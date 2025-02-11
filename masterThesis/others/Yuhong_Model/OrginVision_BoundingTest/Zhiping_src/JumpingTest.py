from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt
import numpy as np
import math

from foreLeg import ForeLegM
from hindLeg import HindLegM

RUN_STEPS = 100

model = load_model_from_path("../models/dynamic_4l_t3.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()
sim.set_state(sim_state)

ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0,-1.2, 0,0,0,0]

fl_params = {'lr0':0.033, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0295, 
    'l2': 0.0145, 'l3': 0.0225, 'l4': 0.0145,'alpha':23*np.pi/180}
fl_left = ForeLegM(fl_params)
fl_right = ForeLegM(fl_params)
# --------------------------------------------------------------------- #
hl_params= {'lr0':0.032, 'rp': 0.008, 'd1': 0.0128, 'l1': 0.0317, 
    'l2': 0.02, 'l3': 0.0305, 'l4': 0.0205,'alpha':73*np.pi/180}
hl_left = HindLegM(hl_params)
hl_right = HindLegM(hl_params)

# idle motion
for i in range(500):  
    sim.data.ctrl[:] = ctrlData
    sim.step()
    viewer.render()

# ctrldata computation
def computer_ctrldata(fp, hp):
    foreLeg_left_q = fl_left.pos_2_angle(0, fp)
    foreLeg_right_q = fl_right.pos_2_angle(0, fp)
    hindLeg_left_q = hl_left.pos_2_angle(0, hp)
    hindLeg_right_q = hl_right.pos_2_angle(0, hp)
    ctrlData = []
    ctrlData.extend(foreLeg_left_q)
    ctrlData.extend(foreLeg_right_q)
    ctrlData.extend(hindLeg_left_q)
    ctrlData.extend(hindLeg_right_q)
    for i in range(4):
        ctrlData.append(0)
    
    return ctrlData

# init motion
ifp = -0.020
ihp = -0.025
ctrlData = computer_ctrldata(ifp, ihp)
for i in range(100): 
    sim.data.ctrl[:] = ctrlData
    sim.step()
    viewer.render()

# run motion
for i in range(RUN_STEPS):
    range_run = 300
    for j in range(range_run):

        if j < (range_run/6):
            fp = (0.020-0.05) / (range_run/6) * j - 0.020
            hp = (0.025-0.055) / (range_run/6) * j - 0.025
            ctrlData = computer_ctrldata(fp, hp)   
        else: 
            # fp = (-0.010+0.05) / (range_run/2) * (range_run-j) - 0.05
            # hp = (-0.015+0.055) / (range_run/2) * (range_run-j) -0.055
            # ctrlData = computer_ctrldata(fp, hp)
            ctrlData = computer_ctrldata(ifp, ihp)

        sim.data.ctrl[:] = ctrlData
        sim.step()
        viewer.render()
