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

        f_int_global_element = truss_nl.CalculateInternalForces(ListOfElement[i][1],ListOfElement[i][2],nodes,disp)

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
        res_norm += res[i,0]**2
    res_norm = np.sqrt(res_norm)  
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
    numerical_limit = 10.0**(-16)
    number_bc = len(ListOfBc)
    for i in range(number_bc):
        current_dof =  ListOfBc[i][0]
        current_disp = ListOfBc[i][1]    #### !! think about initial displacements in case of dynamics --> moving also consider res
        if (abs(current_disp) < numerical_limit):  residual[current_dof] = 0.00



def PrintSolverUpdate(res,res_norm,step):
    print('---------------------------------------------')
    print('step: ', step, ' res_norm: ', res_norm)

def DivideVectorEntries(VecA,VecB):
    numerical_limit = 10.0**(-16)
    system_size = VecA.shape[0]
    Vec_return = np.zeros((system_size,1))
    if (system_size != VecB.shape[0]):
        print('Error! Vector Size does not match!!!') 
    else:
        for i in range(system_size):
            if (abs(VecB[i,0]) > numerical_limit): Vec_return[i,0] = VecA[i,0]/VecB[i,0]
            else: Vec_return[i,0] = 0.00
    return Vec_return

def MultiplyVectorEntries(VecA,VecB):
    system_size = VecA.shape[0]
    Vec_return = np.zeros((system_size,1))
    if (system_size != VecB.shape[0]):
        print('Error! Vector Size does not match!!!') 
    else:
        for i in range(system_size):
            Vec_return[i,0] = VecA[i,0]*VecB[i,0]
    return Vec_return

def CreateInitialForceVector(F_vec,ListOfBc):
    numerical_limit = 10.0**(-16)
    number_Bc = len(ListOfBc)
    F_init = np.zeros((F_vec.shape[0],1))
    for i in range(F_vec.shape[0]):
        F_init[i] = 1.00
    for j in range(number_Bc):
        current_dof =  ListOfBc[i][0]
        current_disp = ListOfBc[i][1] 
        if (abs(current_disp) < numerical_limit): F_init[current_dof] = 0.00  #### !! think about initial displacements in case of dynamics --> moving also consider res
    return F_init

