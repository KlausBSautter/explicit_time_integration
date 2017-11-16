import numpy as np



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

    dxi,dyi,dxj,dyj = disp[dofAi], disp[dofAj], disp[dofBi], disp[dofBj]

    xA,xB,yA,yB = nodeA_ref[1]+dxi,nodeB_ref[1]+dxj,nodeA_ref[2]+dyi,nodeB_ref[2]+dyj
    dx,dy = xB-xA, yB-yA
    l = np.sqrt((dx**2)+(dy**2))
    return dx,dy,l

def CalculateGreenLagrangeStrain(nodes,disp):
    L = CalculateRefLength(nodes)
    dx,dy,l = CalculateCurrentLength(nodes,disp)
    e_gl = 0.50 * (((l*l) / (L*L)) - 1.00)
    return e_gl
    

def ElementStiffMatrix(E,A,nodes,alpha,disp):
    L = CalculateRefLength(nodes)
    e_gl = CalculateGreenLagrangeStrain(nodes,disp)
    dx,dy,l = CalculateCurrentLength(nodes,disp)

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




