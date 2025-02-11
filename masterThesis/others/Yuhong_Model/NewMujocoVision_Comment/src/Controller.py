import numpy as np
import math

from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.legs import LegModel

class MouseController(object):
	"""docstring for MouseController"""
	def __init__(self, fre, time_step, spine_angle):
		super(MouseController, self).__init__()
		PI = np.pi
		#self.curStep = 0# Spine
		self.curRad = 0 # Phase of the legs in footend trajectory
		self.pathStore = LegPath() # Class for the generation of footend trajectory
		# [LF, RF, LH, RH]
		# --------------------------------------------------------------------- #
		#self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
		#self.period = 3/2
		#self.SteNum = 36							#32 # Devide 2*PI to multiple steps
		#self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		self.phaseDiff = [0, PI, PI, 0]			# Trot # difference phasen between four legs
		self.period = 2/2 # time for the swing phase in a cycle
		self.fre_cyc = fre # gait frequence, number of robot steps in 1s
		self.SteNum = int(1/(time_step*self.fre_cyc)) # number of timesteps in a robot step
		print("----> ", self.SteNum)
		self.spinePhase = self.phaseDiff[3] # combine spine phase with RH leg phase
		# --------------------------------------------------------------------- #
		self.spine_A = 2*spine_angle # 10 a_s = 2theta_s
		print("angle --> ", spine_angle) # self.spine_A)
		self.spine_A = self.spine_A*PI/180 # spine angle range, from degree to rad
		# --------------------------------------------------------------------- #
		leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035] # leg length
		self.fl_left = LegModel(leg_params) # define every leg, for forward&inverse kinematic
		self.fl_right = LegModel(leg_params)
		self.hl_left = LegModel(leg_params)
		self.hl_right = LegModel(leg_params)
		# --------------------------------------------------------------------- #
		self.stepDiff = [0,0,0,0] # line 40-43, from phase difference to timestep difference for 4 legs
		for i in range(4):
			self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i]/(2*PI))
		self.stepDiff.append(int(self.SteNum * self.spinePhase/(2*PI)))
		self.trgXList = [[],[],[],[]] # list for desired relativ x position between footend&shoulder joint
		self.trgYList = [[],[],[],[]] # same as above, in y-koordination

	def getLegCtrl(self, leg_M, curRad, leg_ID): # get servo angle for every leg
		leg_flag = "F" # leg_ID [0,3], [0,1] for "F"ront legs, [2,3] for "H"int legs
		if leg_ID > 1:
			leg_flag = "H"
		currentPos = self.pathStore.getOvalPathPoint(curRad, leg_flag, self.period) # compute desired relativ x position between footend&shoulder joint in current timestep
		trg_x = currentPos[0] # relativ position in x-coordination
		trg_y = currentPos[1] # ... in y-...
		self.trgXList[leg_ID].append(trg_x)
		self.trgYList[leg_ID].append(trg_y)
		qVal = leg_M.pos_2_angle(trg_x, trg_y) # from footend position to servo angles, inverse kinematic
		return qVal

	def getSpineVal(self, curRad):
		cur_phase = curRad-self.spinePhase # line 60-62 not relevant right now
		left_spine_val = self.spine_A*(math.cos(cur_phase) + 1)
		right_spine_val = self.spine_A*(math.cos(cur_phase) - 1)
		#return left_spine_val + right_spine_val
		return self.spine_A*math.cos(curRad-self.spinePhase) # spine servo angle
		#spinePhase = 2*np.pi*spineStep/self.SteNum
		#return self.spine_A*math.sin(spinePhase)

	def runStep(self): # line69-76 for leg servo angles computation, line78-79 for spine ...
		foreLeg_left_q = self.getLegCtrl(self.fl_left, 
			self.curRad + self.phaseDiff[0], 0)
		foreLeg_right_q = self.getLegCtrl(self.fl_right, 
			self.curRad + self.phaseDiff[1], 1)
		hindLeg_left_q = self.getLegCtrl(self.hl_left, 
			self.curRad + self.phaseDiff[2], 2)
		hindLeg_right_q = self.getLegCtrl(self.hl_right, 
			self.curRad + self.phaseDiff[3], 3)

		spineRad = self.curRad
		spine_q = self.getSpineVal(spineRad)
		#spine = 0
		step_rad = 0
		if self.SteNum != 0: 
			step_rad = 2*np.pi/self.SteNum # change of rad between two timesteps
		self.curRad += step_rad # update the current phase
		if self.curRad > 2*np.pi:
			self.curRad -= 2*np.pi # if current phase bigger than a cycle phase (2*pi), then minus 2*pi
		ctrlData = [] #line87-100, compute the servo angles for all legs and spine


		#foreLeg_left_q = [1,0]
		#foreLeg_right_q = [1,0]
		#hindLeg_left_q = [-1,0]
		#hindLeg_right_q = [-1,0]
		ctrlData.extend(foreLeg_left_q)
		ctrlData.extend(foreLeg_right_q)
		ctrlData.extend(hindLeg_left_q)
		ctrlData.extend(hindLeg_right_q)
		for i in range(3):
			ctrlData.append(0)
		ctrlData.append(spine_q)
		return ctrlData
		