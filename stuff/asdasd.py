import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def _calculate_length(dx,dy,du,dv):
    x = dx+du
    y = dy+dv
    l2 = (x**2)+(y**2)
    
    return np.sqrt(l2)

def _stiffness_matrix(E,A,L,dv,dy):
    K1 = (E*A)/(2*L*L*L)
    K2 = 3*dv*dv+6*dy*dv+2*dy*dy
    return (K1*K2)

def _dr_dlambda(F):
    return F
def _dc_dv(dv):
    return (2*dv)
def _dc_dlambda(d_lambda):
    return (2*d_lambda)


def _create_solution_matrix(K,drdla,dcdv,dcdla):
    M = [[K,drdla],[dcdv,dcdla]]
    return M

def _calculate_residual(E,A,L,dv,dy,F,lambda_i):
    r1 = (E*A)/(2*L*L*L)
    r2 = dv*dv*dv+3*dy*dv*dv+2*dy*dy*dv
    r3 = F*lambda_i
    return ((r1*r2)+r3)



#costum values
dx = 2
dy = 1
E = 210.00E+009
F = -2.4*(10**7)
A = 0.01
dv = 0
du = 0

L = _calculate_length(dx,dy,du,dv)


##load control
lambda_i = 1.00
d_lambda_i = 0.0001
e_resi = 10000
e_toll = 1.00E-06
d_arc = 0.1
v_array = []
f_array = []

#loop over displacement
e_resi = _calculate_residual(E,A,L,dv,dy,F,lambda_i)

#loop NR
while (abs(e_resi)>e_toll):
    
    M = _create_solution_matrix(_stiffness_matrix(E,A,L,dv,dy),0.00,0.00,1.00)
    print(M)
    M_inv = inv(M)  
    N = [-_calculate_residual(E,A,L,dv,dy,F,lambda_i),0.00]
    print(N)
    P = np.dot(M_inv,N)
    print(P)
    
    dv = dv + P[0]
    print(dv)
    print('####################################')
    e_resi = _calculate_residual(E,A,L,dv,dy,F,lambda_i)
    #print(e_resi)
print(dv)
