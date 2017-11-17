import numpy as np
import truss_element_linear_functions as truss_linear



def CalculateRefLength(nodes):
    nodeA, nodeB = nodes[0],nodes[1]
    dx = nodeB[1] - nodeA[1]
    dy = nodeB[2] - nodeA[2]
    L = np.sqrt((dx**2)+(dy**2))
    return L

def CalculateCurrentLength(nodes,disp):
    nodeA_ref, nodeB_ref = nodes[0],nodes[1]
    dofAi, dofBi = (nodeA_ref[0]-1)*2, (nodeB_ref[0]-1)*2
    dofAj, dofBj = dofAi+1, dofBi+1

    dxi,dyi,dxj,dyj = disp[dofAi,0], disp[dofAj,0], disp[dofBi,0], disp[dofBj,0]

    xA,xB,yA,yB = nodeA_ref[1]+dxi,nodeB_ref[1]+dxj,nodeA_ref[2]+dyi,nodeB_ref[2]+dyj
    dx,dy = xB-xA, yB-yA
    l = np.sqrt((dx**2)+(dy**2))
    return dx,dy,l

def CalculateGreenLagrangeStrain(nodes,disp):
    L = CalculateRefLength(nodes)
    dx,dy,l = CalculateCurrentLength(nodes,disp)
    e_gl = 0.50 * (((l*l) / (L*L)) - 1.00)
    return e_gl
    

def ElementStiffMatrix(E,A,nodes,disp):
    L = CalculateRefLength(nodes)
    e_gl = CalculateGreenLagrangeStrain(nodes,disp)
    dx, dy, l = CalculateCurrentLength(nodes,disp)

    # K_sigma 
    K_sig = np.matrix([[1,0,-1,0],
                       [0,1,0,-1],
                       [-1,0,1,0],
                       [0,-1,0,1]])
    K_sig = K_sig * (E*A*e_gl/L)


    # K0+Ku
    K_ou = np.matrix([[dx*dx,dx*dy,-dx*dx,-dx*dy],
                     [dx*dy,dy*dy,-dx*dy,-dy*dy],
                     [-dx*dx,-dx*dy,dx*dx,dx*dy],
                     [-dx*dy,-dy*dy,dx*dy,dy*dy]])
    K_ou = K_ou * E * A / (L**3)

    K = K_sig+K_ou
    return K


def CalculateInternalForces(E,A,nodes,disp):
    dx,dy,l = CalculateCurrentLength(nodes,disp)
    L = CalculateRefLength(nodes)
    e_gl = CalculateGreenLagrangeStrain(nodes,disp)
    N = (E*A*l*e_gl / L)
    f_int_loc = np.matrix([[-N,0.00,N,0.00]]).T
    
    # rotate to global system
    alpha = np.arctan(dy/dx)
    c = np.cos(alpha)
    s = np.sin(alpha)
    T = np.matrix([[c,-s,0,0],[s,c,0,0],[0,0,c,-s],[0,0,s,c]])
    f_int_glob = np.dot(T,f_int_loc)

    return f_int_glob





