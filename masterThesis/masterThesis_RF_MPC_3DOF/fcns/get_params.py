import numpy as np

def get_params(gait):
    p = {}
    p['predHorizon'] = 6                # MPC Predicted Horizion/Timesteps
    p['simTimeStep'] = 1/2000           # Simulation Timestep set in Mujoco xml-Files
    p['Tmpc'] = 4/1000                  # Time value of one Horizion
    p['gait'] = gait                    # Selected Gait Index
    p['Umax'] = 100                     # Maximal optimal GRF after MPC / System Input Vector
    p['decayRate'] = 1                  # Gamma value in MPC cost function
    p['freq'] = 30                      # Not relevant; for Animation in Matlab
    p['Rground'] = np.eye(3)            # Ground slope
    p['Qf'] = np.diag([1e5, 2e5, 3e5, 5e2, 1e3, 150, 1e3, 1e4, 800, 40, 40, 10]) # Gain value in MPC cost function

    # ---- gait ----
    if gait == 1:           # 1 - bound
        p['Tst'] = 0.1                  # Stance time of one leg in a gait cycle
        p['Tsw'] = 0.18                 # Swing time ...
        p['predHorizon'] = 7
        p['simTimeStep'] = 1/1000
        p['Tmpc'] = 2/1000
        p['decayRate'] = 1
        p['R'] = np.diag(np.squeeze(np.tile(np.array([[0.1, 0.1, 0.1]]).T, (4, 1)).T)) # Gain value in MPC cost function
        p['Q'] = np.diag([5e4, 2e4, 1e6, 4e3, 5e2, 5e2, 1e4, 5e4, 1e3, 1e2, 5e2, 1e2]) # Gain value in MPC cost function
        p['Qf'] = np.diag([2e5, 5e4, 5e6, 8e3, 5e2, 5e2, 1e4, 5e4, 5e3, 1e2, 1e2, 1e2]) # Gain value in MPC cost function
    elif gait == 2:         # 2 - pacing
        p['Tst'] = 0.12
        p['Tsw'] = 0.12
        p['R'] = np.diag(np.squeeze(np.tile(np.array([[0.1, 0.2, 0.1]]).T, (4, 1)).T))
        p['Q'] = np.diag([5e3, 5e3, 9e4, 5e2, 5e2, 5e2, 7e3, 7e3, 7e3, 5e1, 5e1, 5e1])
    elif gait == 3:         # 3 - gallop
        p['Tst'] = 0.08
        p['Tsw'] = 0.2
        p['R'] = np.diag(np.squeeze(np.tile(np.array([[0.1, 0.2, 0.1]]).T, (4, 1)).T))
        p['Q'] = np.diag([3e3, 3e3, 4e6, 5e2, 1e3, 150, 1e4, 1e4, 800, 1e2, 5e1, 5e1])
    elif gait == 4:         # 4 - trot run
        p['Tst'] = 0.12
        p['Tsw'] = 0.2
        p['Tmpc'] = 3/1000
        p['predHorizon'] = 6
        p['decayRate'] = 1
        p['R'] = np.diag(np.squeeze(np.tile(np.array([[0.1, 0.18, 0.08]]).T, (4, 1)).T))
        p['Q'] = np.diag([1e5, 1e5, 1e5, 1e3, 1e3, 1e3, 2e3, 1e4, 800, 100, 40, 10])
        p['Qf'] = np.diag([1e5, 1.5e5, 2e4, 1.5e3, 1e3, 100, 2e3, 2e3, 800, 100, 60, 10])
    elif gait == 5:         # 5 - crawl
        p['Tst'] = 0.3
        p['Tsw'] = 0.1
        p['R'] = np.diag(np.squeeze(np.tile(np.array([[0.1, 0.2, 0.1]]).T, (4, 1)).T))
        p['Q'] = np.diag([5e5, 5e5, 9e5, 5, 5, 5, 3e3, 3e3, 3e3, 3, 3, 3])
    else:                   # 0 - trot
        p['Tst'] = 0.3
        p['Tsw'] = 0.15
        p['predHorizon'] = 6
        p['simTimeStep'] = 1/1000
        p['Tmpc'] = 8/1000
        p['R'] = np.diag(np.squeeze(np.tile(np.array([[0.1, 0.2, 0.1]]).T, (4, 1)).T))
        p['Q'] = np.diag([1e5, 2e5, 3e5, 5e2, 1e3, 1e3, 1e3, 1e4, 800, 40, 40, 10])
        p['Qf'] = p['Q']

    ## Physical Parameters
    p['mass'] = 5.5                  # Body mass of the robot
    p['J'] = np.diag([4.695e-03, 7.632e-02, 7.873e-02]) # Body moment of inertia in [x,y,z]
    # p['J'] = np.diag([0.026, 0.112, 0.075])
    p['g'] = 9.81                    # Gravity Acceleration
    p['mu'] = 1                      # friction coefficient
    p['z0'] = 0.2                    # nominal COM height
    p['pf34'] = np.array([[0.15, 0.094, 0], [0.15, -0.094, 0], [-0.15, 0.094, 0], [-0.15, -0.094, 0]]).T
    p['L'] = 0.301                   # body length
    p['W'] = 0.088                   # body width
    p['d'] = 0.05                    # ABAD offset
    p['h'] = 0.05                    # body height
    p['l1'] = 0.14                   # link1 length
    p['l2'] = 0.14                   # link2 length

    p['J_UD_Leg'] = np.array([1.045e-04, 1.045e-04, 2.75e-06]) # Up/Down Leg Inertia; Used for leg dynamics; Not relevant right now
    p['m_UpLeg'] = 0.055             # Up Leg Mass
    p['m_DownLeg'] = 0.055           # Down Leg Mass

    ## Swing phase
    p['Kp_sw'] = np.array([1600, 1600, 1600])     # Kp for swing phase
    p['Kd_sw'] = np.array([5, 5, 5])              # Kd for swing phase
    p['Kp_init'] = np.array([1000, 500, 500])     # Kp for init stance phase
    p['Kd_init'] = np.array([80, 40, 40])         # Kd for init stance phase

    ## Color (Not Relevant; For Plotting in Matlab)
    p['body_color'] = np.array([42/255, 80/255, 183/255])
    p['leg_color'] = np.array([7/255, 179/255, 128/255])
    p['ground_color'] = np.array([195/255, 232/255, 243/255])

    return p
