import numpy as np
import math

def ry(theta):
      c = math.cos(theta)
      s = math.sin(theta)

      Ry = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
      
      return Ry