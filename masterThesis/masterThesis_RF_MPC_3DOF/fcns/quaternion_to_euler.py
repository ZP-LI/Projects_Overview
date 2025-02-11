import numpy as np
import math

# Transform Quaternion to Euler angle

def quaternion_to_euler(Quat): 
    w = Quat[0]
    x = Quat[1]
    y = Quat[2]
    z = Quat[3]

    t0 = + 2.0 * (w * x + y * z)
    t1 = + 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = + 2.0 * (w * y - z * x)
    t2 = + 1.0 if t2 > +1.0 else t2
    t2 = - 1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = + 2.0 * (w * z + x * y)
    t4 = + 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])