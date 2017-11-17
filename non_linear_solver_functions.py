import numpy as np 
import truss_element_linear_functions as truss
import truss_element_non_linear_functions as truss_nl


def AssembleInternalForceVector(ListOfElement,disp,system_size):
    number_elements = len(ListOfElement)
    f_int_global_total = np.zeros((system_size,1))

    for i in range(number_elements):
        nodes = ListOfElement[i][3]
        nodeA_ref, nodeB_ref = nodes[0],nodes[1]
        dofAi, dofBi = (nodeA_ref[0]-1)*2, (nodeB_ref[0]-1)*2
        dofAj, dofBj = dofAi+1, dofBi+1

        disp_i = np.matrix([[disp[dofAi,0],disp[dofAj,0],disp[dofBi,0],disp[dofBj,0]]]).T
        f_int_global_element = truss_nl.CalculateInternalForces(ListOfElement[i][1],ListOfElement[i][2],nodes,disp_i)

        f_int_global_total[dofAi,0] += f_int_global_element[0,0]
        f_int_global_total[dofAj,0] += f_int_global_element[1,0]
        f_int_global_total[dofBi,0] += f_int_global_element[2,0]
        f_int_global_total[dofBj,0] += f_int_global_element[3,0]
    return f_int_global_total

def CalculateResidualStatic(F_int,F_ext):
    return (F_int-F_ext)


def ResidualNorm(res):
    system_size = res.shape[0]
    res_norm = 0.00
    for i in range(system_size):
        res_norm += res[i][0]**2
    res_norm = np.sqrt(res_norm[0,0])  
    return res_norm

def UpdateStiffnessMatrix(ListOfElement,disp,ListOfBc):
    number_elements = len(ListOfElement)
    new_elements = []
    for i in range(number_elements):
        nodes = ListOfElement[i][3]
        E,A =   ListOfElement[i][1],ListOfElement[i][2]
        new_elements.append([truss_nl.ElementStiffMatrix(E,A,nodes,disp),E,A,nodes])

    K_master = truss.AssembleElementMatrices(new_elements)
    F_master = np.zeros((K_master.shape[0],1))
    K_mod, F_mod = truss.ModifyMasterMatrix(K_master,F_master,ListOfBc)

    return K_mod

def ModifyResidual(residual,ListOfBc):
    number_bc = len(ListOfBc)
    for i in range(number_bc):
        current_dof =  ListOfBc[i][0]
        residual[current_dof] = 0.00