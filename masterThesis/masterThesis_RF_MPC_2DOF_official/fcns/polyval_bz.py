# Function to evaluate bezier polynomials
# Inputs: alpha - Bezeir coefficients - (alpha_0,) or (multiple num - "m", alpha_"m")
#         s - s parameter. Range [0 1] - (single float num) or (multiple num - "m",)
# Outputs: b = sum(k=0 to m)[ alpha_k * M!/(k!(M-k)!) s^k (1-s)^(M-k)] - (single float num) or (multiple num - "m",)

import numpy as np

def polyval_bz(alpha, s):
    try:
        s.shape[1]
    
    except: # single input
        b = 0
        M = alpha.shape[0] - 1

        for k in range(M+1):
            b += alpha[k] * np.math.factorial(M) / (np.math.factorial(k) * np.math.factorial(M-k)) * np.power(s, k) * np.power(1-s, M-k)
    
    else: # multiple inputs
        b = np.zeros(s.shape)
        M = alpha.shape[1] - 1
        
        for k in range(M+1):
            b += alpha[:, k] * np.math.factorial(M) / (np.math.factorial(k) * np.math.factorial(M-k)) * np.power(s, k) * np.power(1-s, M-k)
    
    return b