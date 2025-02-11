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
		self.curRad = 0
		self.pathStore = LegPath()
		# [LF, RF, LH, RH]
		# --------------------------------------------------------------------- #
		#self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
		#self.period = 3/2
		#self.SteNum = 36							#32 # Devide 2*PI to multiple steps
		#self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		self.phaseDiff = [0, PI, PI, 0]			# Trot
		self.period = 2/2
		self.fre_cyc = fre
		self.SteNum = int(1/(time_step*self.fre_cyc)) # 0.5步/秒，0.002秒/timestep -> SteNum = 几个tiemstep/步
		print("----> ", self.SteNum)
		self.spinePhase = self.phaseDiff[3]
		# --------------------------------------------------------------------- #
		self.spine_A = 2*spine_angle # 10 a_s = 2theta_s
		print("angle --> ", spine_angle) # self.spine_A)
		self.spine_A = self.spine_A*PI/180
		# --------------------------------------------------------------------- #
		leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
		self.fl_left = LegModel(leg_params)
		self.fl_right = LegModel(leg_params)
		self.hl_left = LegModel(leg_params)
		self.hl_right = LegModel(leg_params)
		# --------------------------------------------------------------------- #
		self.stepDiff = [0,0,0,0]
		for i in range(4):
			self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i]/(2*PI))
		self.stepDiff.append(int(self.SteNum * self.spinePhase/(2*PI)))
		self.trgXList = [[],[],[],[]]
		self.trgYList = [[],[],[],[]]

	def getLegCtrl(self, leg_M, curRad, leg_ID):
		leg_flag = "F"
		if leg_ID > 1:
			leg_flag = "H"
		currentPos = self.pathStore.getOvalPathPoint(curRad, leg_flag, self.period)
		trg_x = currentPos[0]
		trg_y = currentPos[1]
		self.trgXList[leg_ID].append(trg_x)
		self.trgYList[leg_ID].append(trg_y)
		qVal = leg_M.pos_2_angle(trg_x, trg_y)
		return qVal

	def getSpineVal(self, curRad):
		cur_phase = curRad-self.spinePhase
		left_spine_val = self.spine_A*(math.cos(cur_phase) + 1)
		right_spine_val = self.spine_A*(math.cos(cur_phase) - 1)
		#return left_spine_val + right_spine_val
		return self.spine_A*math.cos(curRad-self.spinePhase)
		#spinePhase = 2*np.pi*spineStep/self.SteNum
		#return self.spine_A*math.sin(spinePhase)

	def runStep(self):
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
		if self.SteNum != 0: # 这里验证SteNum与0的关系有什么意义？SteNum什么时候会等于0？
			step_rad = 2*np.pi/self.SteNum
		self.curRad += step_rad
		if self.curRad > 2*np.pi:
			self.curRad -= 2*np.pi
		ctrlData = []


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
		