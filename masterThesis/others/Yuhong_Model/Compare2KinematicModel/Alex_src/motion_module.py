#!/usr/bin/env python3
# *******************************************************
# Type: Motion controller
# 
# Motion controller for the mouse
# Handles state machine and low level spine and leg control.
# 
# Author: Alex Rohregger
# Contact: alex.rohregger@tum.de
# Last-edited: 26.04.2021
# *********************************************************

# Import the leg and motionplanner modules
from mouse_controller.leg_controller import Leg_Controller
from mouse_controller.state_machine.leg_state_machine import Leg_State_Machine
from mouse_controller.mouse_parameters_dir import Gait_Parameters, Mouse_Parameters
from time import sleep

# Import other relevant libraries for ROS
import numpy as np
import math
from time import sleep
import time

from numpy.core.numerictypes import maximum_sctype

from mujoco_py import load_model_from_path, MjSim, MjViewer
import matplotlib.pyplot as plt

class Motion_Module:
    def __init__(self):
        self.model_name = "dynamic_4l_t3.xml"
        self.model_path = "../models/"+self.model_name
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        # self.fixPoint = "body_ss"
        self.fixPoint = "imu"
        self.movePath = [[],[],[]]
        self.legPosName = [
            ["router_shoulder_fl", "foot_s_fl"],
            ["router_shoulder_fr", "foot_s_fr"],
            ["router_hip_rl", "foot_s_rl"],
            ["router_hip_rr", "foot_s_rr"]]
        self.legRealPoint_x = [[],[],[],[]]
        self.legRealPoint_y = [[],[],[],[]]
        self.trgXList = [[],[],[],[]]
        self.trgYList = [[],[],[],[]]
        self.time_cost = 0

        self.init_mouse_variables()
        self.init_controllers()
        self.main()

    
    def init_mouse_variables(self):
        self.gait_parameters2 = Gait_Parameters()
        self.mouse_parameters = Mouse_Parameters()
        self.general_st_parameters2 = self.gait_parameters2.st_trot_parameters
        self.front_leg_parameters2 = self.gait_parameters2.st_trot_param_f
        self.rear_leg_parameters2 = self.gait_parameters2.st_trot_param_r

    def init_controllers(self):
        # Initialize the key components of the motion module
        # Spine modes:
        # 0: purely turning motion, nothing else
        # 1: turning motion + spine modulation
        # 2: turning motion + balance mode (balance mode for 2 and 3 leg contact)
        self.spine_mode = 0
        self.offset_mode = False
        self.balance_mode = True
        self.fsm = Leg_State_Machine(self.general_st_parameters2)
        self.leg_controller = Leg_Controller(self.gait_parameters2, self.mouse_parameters)
        self.vel_in = 0.0
        self.turn_rate = 0.0
        self.buttons = [0]*4
        self.prev_buttons = [0]*4
        self.leg_n = 0

    def main(self):
        self.motion_node()

        start_p = [self.movePath[0][0], self.movePath[1][0]]
        end_p = [self.movePath[0][-1], self.movePath[1][-1]]
        dis = math.sqrt((end_p[0]-start_p[0])*(end_p[0]-start_p[0])
            + (end_p[1]-start_p[1])*(end_p[1]-start_p[1]))
        print(">>>>>>>>>>>>>>>>>>>>>>>>: ", self.time_cost)
        print("Py_v: ", dis/self.time_cost)
        ideal_time_cost = 10000*0.002
        print("Sim_v: ", dis/ideal_time_cost)

    def motion_node(self):
        # main starter method
        q_legs = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0, -1.2]
        q_values = np.concatenate((q_legs,np.array(([0,0,0,0,0]))))
        for i in range(500):
            self.sim.data.ctrl[:] = q_values
            self.sim.step()
            self.viewer.render() 

        self.fsm.timer.reset_times()
        sleep(0.002)
        self.offset_mode = True
        #self.leg_n = (self.leg_n+1)%4
        start = time.time()
        tTime = 0
        for i in range(10000):
            self.vel_in = 0.4
            self.turn_rate = 0
            vel = self.vel_in * np.ones((4,))
            
            leg_states, leg_timings, norm_time = self.fsm.run_state_machine()
            # Steps of the full controller to generate values
            
            target_leg_positions, q_legs, q_spine = \
                self.leg_controller.run_controller(leg_states, leg_timings, 
                    norm_time, vel, self.turn_rate, self.spine_mode, self.offset_mode)
            
            q_tail = 0 # self.tail_extension(norm_time, self.vel_in)
            tStart = time.time()
            self.gen_messages(target_leg_positions, q_legs, q_spine, q_tail)
            tEnd = time.time()
            tTime += tEnd - tStart

        end = time.time()
        self.time_cost = end - start
        # print("Traj --> ", self.leg_controller.traj_time)
        # print("Spine --> ", self.leg_controller.sVal_time)
        # print("Leg --> ", self.leg_controller.qVal_time)
        total_t = self.leg_controller.traj_time + self.leg_controller.sVal_time + \
        self.leg_controller.qVal_time
        # print("Cal --> ", self.time_cost-total_t)
        # print(tTime)
        # """
        with open('TestAlexModel_125_TestYuhongTraj_Li.txt', 'w') as f:
            f.write('Body Position:')
            f.write('\n')
            for i in range(len(self.movePath[0])):
                f.write(','.join([str(self.movePath[0][i]), str(self.movePath[1][i]), str(self.movePath[2][i])]))
                f.write('\n')
            f.write('\n')
            f.write('Real and Target EndpointPosition of Leg1 (fl):')
            f.write('\n')
            for i in range(len(self.legRealPoint_x[0])):
                f.write(','.join([str(self.trgXList[0][i]), str(self.trgYList[0][i])]))
                f.write('   ')
                f.write(','.join([str(self.legRealPoint_x[0][i]), str(self.legRealPoint_y[0][i])]))
                f.write('\n')
            f.write('\n')
            f.write('Real and Target EndpointPosition of Leg2 (fr):')
            f.write('\n')
            for i in range(len(self.legRealPoint_x[0])):
                f.write(','.join([str(self.trgXList[1][i]), str(self.trgYList[1][i])]))
                f.write('   ')
                f.write(','.join([str(self.legRealPoint_x[1][i]), str(self.legRealPoint_y[1][i])]))
                f.write('\n')
            f.write('\n')
            f.write('Real and Target EndpointPosition of Leg3 (rl):')
            f.write('\n')
            for i in range(len(self.legRealPoint_x[0])):
                f.write(','.join([str(self.trgXList[2][i]), str(self.trgYList[2][i])]))
                f.write('   ')
                f.write(','.join([str(self.legRealPoint_x[2][i]), str(self.legRealPoint_y[2][i])]))
                f.write('\n')
            f.write('\n')
            f.write('Real and Target EndpointPosition of Leg4 (rr):')
            f.write('\n')
            for i in range(len(self.legRealPoint_x[0])):
                f.write(','.join([str(self.trgXList[3][i]), str(self.trgYList[3][i])]))
                f.write('   ')
                f.write(','.join([str(self.legRealPoint_x[3][i]), str(self.legRealPoint_y[3][i])]))
                f.write('\n')
            f.write('\n')
            f.write('All kind of Timer:')
            f.write('\n')
            f.write('   Sum Time:')
            f.write(str(self.time_cost))
            f.write('\n')
            f.write('   Traj Time:')
            f.write(str(self.leg_controller.traj_time))
            f.write('\n')
            f.write('   Spine Time:')
            f.write(str(self.leg_controller.sVal_time))
            f.write('\n')
            f.write('   Leg Time:')
            f.write(str(self.leg_controller.qVal_time))
            f.write('\n')
            f.write('   Gen_message Time:')
            f.write(str(tTime))
            f.write('\n')
        # """

    def tail_extension(self, timing: float,vel: float, offset=0, scaling=0.5) -> float:
        # THis function helps extend the spine stride during gait.
        # Timing value: is normalized time value [0,1]
        scale = min(4.5*np.abs(vel)**2, scaling)
        q_tail = scale*np.cos(2*np.pi*timing+offset)
        return q_tail

    def gen_messages(self, target_leg_positions, q_legs, q_spine, q_tail):
        target_leg_positions.astype(dtype=np.float32)
        q_values = np.concatenate((q_legs,np.array(([q_tail,0,0,q_spine,0]))))
        q_values.astype(dtype=np.float32)

        self.sim.data.ctrl[:] = q_values
        self.sim.step()
        self.viewer.render() 

        tData = self.sim.data.get_site_xpos(self.fixPoint)
        for i in range(3):
            self.movePath[i].append(tData[i])

        for i in range(4):
            originPoint = self.sim.data.get_site_xpos(self.legPosName[i][0]) # Position of point O
            currentPoint = self.sim.data.get_site_xpos(self.legPosName[i][1]) # Position of point D or H
            #print(originPoint, currentPoint)
            tX = currentPoint[1]-originPoint[1]
            tY = currentPoint[2]-originPoint[2]
            self.legRealPoint_x[i].append(tX)
            self.legRealPoint_y[i].append(tY)

            self.trgXList[i].append(target_leg_positions[i][0])
            self.trgYList[i].append(target_leg_positions[i][1])

if __name__ == "__main__":
    Motion_Module()
