#!/usr/bin/env python3

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import rospkg
import math
import numpy as np
import random
import sys

#rp = rospkg.RosPack()
#script_path = os.path.join(rp.get_path("mouse_controller"), "models")
#sys.path.append(script_path)
sys.path.append('/home/pipi/catkin_ws/src/mouse_controller')

from numpy.core.numerictypes import maximum_sctype
import pandas as pd
import time
import pathlib
import rospy
import roslib
from std_msgs.msg import Empty, _String

# Import the leg and motionplanner modules
#sys.path.append('/home/pipi/catkin_ws/src/mouse_controller/src/mouse_controller')
#from mouse_controller.leg_unit_class import Leg_Unit
#from mouse_controller.four_legged_body import Quadruped_Walker

# Import model relevant parameters
# load up of the model of front leg type 1
#model_name = "dynamic_4l_t3.xml"
#model_path = os.path.join(rp.get_path("mouse_controller"), "models", model_name)
model_path = '/home/pipi/catkin_ws/src/mouse_controller/models/dynamic_4l_t3.xml'
model = load_model_from_path(model_path)
# model = load_model_from_path("../models/realistic_leg_1_jumper.xml")
sim = MjSim(model)

# initialize viewer for rendering
# viewer = MjViewer(sim)


# Number of test runs to get the kinematics
test_runs = 100000
dead_time = 20


# Allowed range of the motors
m1_range = {"max": 1.0,
            "min": -1.57}
m2_range = {"max": 1.4,
            "min": -2.0}


sim_state = sim.get_state()

sim.set_state(sim_state)

# initialize viewer for rendering
viewer = MjViewer(sim)

for i in range(dead_time):
    sim.data.ctrl[:] = np.array([0]*13)
    sim.step()
    current_servo_values = sim.data.qpos[:]
    print(current_servo_values)
    viewer.render()

sensordata = sim.data.sensordata
print(sensordata)