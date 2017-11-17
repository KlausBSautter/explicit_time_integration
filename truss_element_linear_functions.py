import numpy as np
import copy
import general_functions as general

def CalculateRefLength(nodes):
    nodeA, nodeB = nodes[0],nodes[1]
    dx = nodeB[1] - nodeA[1]
    dy = nodeB[2] - nodeA[2]
    L = np.sqrt((dx**2)+(dy**2))
    return L

def CalculateInclinationAngle(nodes):
    nodeA, nodeB = nodes[0],nodes[1]
    dx = nodeB[1] - nodeA[1]
    dy = nodeB[2] - nodeA[2]
    alpha = np.arctan(dy/dx)
    return alpha

def ElementStiffMatrix(E,A,nodes):
    alpha = CalculateInclinationAngle(nodes)
    c = np.cos(alpha)
    s = np.sin(alpha)
    L = CalculateRefLength(nodes)
    K = np.matrix([[c*c,s*c,-c*c,-s*c],
                   [s*c,s*s,-s*c,-s*s],
                   [-c*c,-s*c,c*c,s*c],
                   [-s*c,-s*s,s*c,s*s]])
    K = K * (E*A/L)
    return K

def ElementMassMatrix(rho,A,nodes):
    L = CalculateRefLength(nodes)
    m = rho*L*A
    M = np.matrix([[m/2.00,0.00,0.00,0.00],
                   [0.00,m/2.00,00,00],
                   [0.00,0.00,m/2.00,0.00],
                   [0.00,0.00,0.00,m/2.00]])
    return M

def MasterDampingMatrix(K,M,ListOfBc,criticalDampingRatio):
    eigenval = general.FindEigenValues(M,K,ListOfBc) 
    wi = eigenval[0]
    wj = wi * 100000
    if (len(eigenval)>1): wj = eigenval[1]
    else: print('Attention! second eigenfrequency not available')
    W = 0.500*np.matrix([[1.00/wi,wi],[1/wj,wj]])
    W = np.linalg.inv(W)
    xi = np.matrix([[criticalDampingRatio,criticalDampingRatio]]).T
    cooef = np.dot(W,xi)
    C = cooef[0,0] * M + cooef[1,0] * K
    return C
    



def AssembleElementMatrices(InputList):
    SystemSize = len(InputList)
    ListOfElements = []
    etab = []
    for i in range(SystemSize):
        ListOfElements.append(InputList[i][0])
        etab.append([InputList[i][3][0][0],InputList[i][3][1][0]])


    number_elements = len(ListOfElements)
    number_nodes = FindMaxEntryInListList(etab)
    number_dofs_element = 2
    number_dofs = number_nodes * number_dofs_element

    # initialize global Matrix
    M_global = np.zeros((number_dofs,number_dofs))
    # loop over all elements
    for i in range(number_elements):
        dof_i_A = (etab[i][0]-1)*number_dofs_element
        dof_i_B = (etab[i][1]-1)*number_dofs_element
        etab_elment = [dof_i_A,dof_i_A+1,dof_i_B,dof_i_B+1]
        K_e = ListOfElements[i]
        if (max(etab_elment)>number_dofs): print('Error: DOF not available !!')
        # loop over element freedom table
        for j in range(len(etab_elment)):
            current_dof_j = etab_elment[j]
            for k in range(len(etab_elment)):
                current_dof_k = etab_elment[k]
                M_global[current_dof_j,current_dof_k] += K_e[j,k]
    return M_global


def FindMaxEntryInListList(ListList):
    max_entry = 0
    for i in range(len(ListList)):
        for j in range(len(ListList[i])):
            if ListList[i][j] > max_entry:
                max_entry = ListList[i][j]
    return max_entry

def ModifyMasterMatrix(MasterMatrix,MasterForce,ListOfBc):
    number_bc = len(ListOfBc)
    number_dof = MasterMatrix.shape[0]

    F_mod = copy.deepcopy(MasterForce)
    K_mod = copy.deepcopy(MasterMatrix)

    for i in range(number_bc):
        current_dof = ListOfBc[i][0]
        current_disp = ListOfBc[i][1]

        for j in range(number_dof):
            F_mod[j] -= MasterMatrix[j,current_dof]*current_disp
            K_mod[j,current_dof] = 0.00
            K_mod[current_dof,j] = 0.00
                
    for i in range(number_bc):
        current_dof = ListOfBc[i][0]
        current_disp = ListOfBc[i][1]
        F_mod[current_dof] = current_disp
        K_mod[current_dof,current_dof] = 1.00        


    return K_mod,F_mod



def CalculateInternalForces(MasterMatrix,current_disp):
    f_int = np.dot(MasterMatrix,current_disp)
    return f_int