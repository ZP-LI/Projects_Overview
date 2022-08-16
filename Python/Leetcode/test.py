import numpy as np
#'''
max_vel = 0.01

# *** Rear Leg Gait Pattern ***
n_r = np.array([0, -0.045])

st_r = np.array([0.04, -0.023])
a_r = np.array([0.04, -0.055])
b_r = np.array([-0.015, -0.04])

s_r = 6

trajectory_r = [st_r,b_r,a_r]
max_step_r = [0.02*s_r, 0.02*s_r, 0.02*s_r]

n_f = np.array([0, -0.045])

st_f = np.array([-0.035, -0.025])
a_f = np.array([0.005, -0.02])
b_f = np.array([0.005, -0.05])

s_f = 6

trajectory_f = [a_f,st_f, b_f]
max_step_f = [0.02*s_f,0.02*s_f,0.02*s_f]

trajectory_rest = [n_f,n_f,n_f]

print(n_r, n_f)
print(trajectory_r, trajectory_f)
print(max_step_r, max_step_f)
print(trajectory_rest)
'''
a = [3, 4, 2, 4, 5, 2]
print(a[4:])
'''