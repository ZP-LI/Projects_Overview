import numpy as np
import math

def rz(psi):
      c = math.cos(psi)
      s = math.sin(psi)

      Rz = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
      
      return Rz