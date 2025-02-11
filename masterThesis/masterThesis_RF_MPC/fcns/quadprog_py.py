import numpy as np
from scipy.optimize import minimize

def quadprog_py(H, g, Aineq, bineq, Aeq, beq, lb=None, ub=None):
    # set "z" be the parameters/vector to be optimized

    # Define the initial guess for z
    z0 = np.zeros_like(g)

    # Define the bounds for z
    # low bounds and upper bounds should have same size as "z"
    try: 
        len(lb)
    except:
        bounds = None # If you don't want to specify bounds, set this to None
    else:
        bounds = [(lb[i], ub[i]) for i in range(len(lb))]

    # Set up the optimization problem and solve it
    res = minimize(obj, z0, args = (H, g), 
                   method = 'SLSQP', 
                   constraints = (
                    {'type': 'ineq', 'fun': ineq_con, 'args': (Aineq, bineq)},
                    {'type': 'eq', 'fun': eq_con, 'args': (Aeq, beq)}), 
                   bounds = bounds)

    # Extract the solution from the optimization result
    zval = res.x

    return zval

# Define the objective function
# min( 1/2 * z^T * H * z + g^T * z )
def obj(z, H, g):
    return 0.5 * z.T @ H @ z + g.T @ z

# Define the inequality constraint function
# A_ineq * z <= b_ineq
# => b_ineq - A_ineq * z >= 0
def ineq_con(z, Aineq, bineq):
    return bineq - Aineq @ z

# Define the equality constraint function
# A_eq * z = b_eq
# => A_eq * z - b_eq = 0
def eq_con(z, Aeq, beq):
    return Aeq @ z - beq