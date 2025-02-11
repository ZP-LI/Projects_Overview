import numpy as np
import math

def rx(phi):
      c = math.cos(phi)
      s = math.sin(phi)

      Rx = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]])
      
      return Rx