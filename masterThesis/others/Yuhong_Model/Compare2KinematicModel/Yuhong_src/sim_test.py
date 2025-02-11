from ToSim import SimModel
from Controller import MouseController
import matplotlib.pyplot as plt
import time


# --------------------
RUN_STEPS = 10000 #320
if __name__ == '__main__':
	theMouse = SimModel("../models/dynamic_4l_t3.xml")

	fre = 1.00 # 1.25/1.00/0.80/0.67
	theController = MouseController(fre)
	for i in range(500):
		ctrlData = [0.0, 1.5, 0.0, 1.5, 0.0, -1.2, 0.0, -1.2, 0, 0, 0, 0, 0]
		theMouse.runStep(ctrlData)
	theMouse.initializing()
	start = time.time()
	for i in range(RUN_STEPS):
		#print("Step --> ", i)
		tCtrlData = theController.runStep()				# No Spine
		#tCtrlData = theController.runStep_spine()		# With Spine
		ctrlData = tCtrlData
		theMouse.runStep(ctrlData)
	end = time.time()
	timeCost = end-start
	total_t = theController.traj_time + theController.sVal_time + theController.qVal_time
	other_t = timeCost - total_t

	# theMouse.movePath
	# theMouse.legRealPoint_x
	# theMouse.legRealPoint_y
	# theController.trgXList
	# theController.trgYList

	'''
	with open('TestYuhongModel_125_TestAlexTraj_Li.txt', 'w') as f:
		f.write('Body Position:')
		f.write('\n')
		for i in range(len(theMouse.movePath[0])):
			f.write(','.join([str(theMouse.movePath[0][i]), str(theMouse.movePath[1][i]), str(theMouse.movePath[2][i])]))
			f.write('\n')
		f.write('\n')
		f.write('Real and Target EndpointPosition of Leg1 (fl):')
		f.write('\n')
		for i in range(len(theMouse.legRealPoint_x[0])):
			f.write(','.join([str(theController.trgXList[0][i]), str(theController.trgYList[0][i])]))
			f.write('   ')
			f.write(','.join([str(theMouse.legRealPoint_x[0][i]), str(theMouse.legRealPoint_y[0][i])]))
			f.write('\n')
		f.write('\n')
		f.write('Real and Target EndpointPosition of Leg2 (fr):')
		f.write('\n')
		for i in range(len(theMouse.legRealPoint_x[0])):
			f.write(','.join([str(theController.trgXList[1][i]), str(theController.trgYList[1][i])]))
			f.write('   ')
			f.write(','.join([str(theMouse.legRealPoint_x[1][i]), str(theMouse.legRealPoint_y[1][i])]))
			f.write('\n')
		f.write('\n')
		f.write('Real and Target EndpointPosition of Leg3 (rl):')
		f.write('\n')
		for i in range(len(theMouse.legRealPoint_x[0])):
			f.write(','.join([str(theController.trgXList[2][i]), str(theController.trgYList[2][i])]))
			f.write('   ')
			f.write(','.join([str(theMouse.legRealPoint_x[2][i]), str(theMouse.legRealPoint_y[2][i])]))
			f.write('\n')
		f.write('\n')
		f.write('Real and Target EndpointPosition of Leg4 (rr):')
		f.write('\n')
		for i in range(len(theMouse.legRealPoint_x[0])):
			f.write(','.join([str(theController.trgXList[3][i]), str(theController.trgYList[3][i])]))
			f.write('   ')
			f.write(','.join([str(theMouse.legRealPoint_x[3][i]), str(theMouse.legRealPoint_y[3][i])]))
			f.write('\n')
		f.write('\n')
		f.write('All kind of Timer:')
		f.write('\n')
		f.write('   Sum Time:')
		f.write(str(timeCost))
		f.write('\n')
		f.write('   Traj Time:')
		f.write(str(theController.traj_time))
		f.write('\n')
		f.write('   Spine Time:')
		f.write(str(theController.sVal_time))
		f.write('\n')
		f.write('   Leg Time:')
		f.write(str(theController.qVal_time))
		f.write('\n')
		f.write('   Gen_message and other Time:')
		f.write(str(other_t))
		f.write('\n')
	'''

	# print("Time -> ", timeCost)
	# print("Traj --> ", theController.traj_time)
	# print("Spine --> ", theController.sVal_time)
	# print("Leg --> ", theController.qVal_time)

	# print('Others --> ', timeCost - total_t)



	dis = theMouse.drawPath()
	print("py_v --> ", dis/timeCost)
	print("sim_v --> ", dis/(RUN_STEPS*0.002))