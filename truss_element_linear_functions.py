import numpy as np
import copy

def ElementStiffMatrix(E,A,L,alpha):
    c = np.cos(alpha)
    s = np.sin(alpha)
    K = np.matrix([[c*c,s*c,-c*c,-s*c],
                   [s*c,s*s,-s*c,-s*s],
                   [-c*c,-s*c,c*c,s*c],
                   [-s*c,-s*s,s*c,s*s]])
    K = K * (E*A/L)
    return K

def ElementMassMatrix(rho,A,L):
    m = rho*L*A
    M = np.matrix([[m/2.00,0.00,0.00,0.00],
                   [0.00,m/2.00,00,00],
                   [0.00,0.00,m/2.00,0.00],
                   [0.00,0.00,0.00,m/2.00]])
    return M


def AssembleElementMatrices(ListOfElements,etab):
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