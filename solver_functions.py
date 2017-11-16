import numpy as np 



def solve_linear(LHS,RHS):
    u = np.linalg.solve(LHS,RHS)
    return u