import numpy as np 
import copy
import explicit_functions as explicit
import truss_element_linear_functions as truss
import truss_element_non_linear_functions as truss_nl



def solve_linear(LHS,RHS):
    u = np.linalg.solve(LHS,RHS)
    return u


#ListOfElement [[element1,E,A,nodes]]
def solve_nonlinear_nr(K_T,ListOfElement,ListOfBc,F_master):

    e_tollerance = 10**(-6)
    system_size = K_T.shape[0]
    K_n = copy.deepcopy(K_T)
    disp_n = explicit.CreateInitialDisplacementVector(ListOfBc,system_size)
    f_int_n = AssembleInternalForceVector(ListOfElement,disp_n,system_size)
    r_n = CalculateResidualStatic(f_int_n,F_master)
    r_n_norm = ResidualNorm(r_n)

    n = 0 ##test
    #while (r_n_norm > e_tollerance):
    while (n < 4):
        n += 1
        K_n_inv = np.linalg.inv(K_n)
        disp_n_1 = np.dot(K_n_inv,r_n)
        disp_n_1 = disp_n - disp_n_1
        print(disp_n_1)

        f_int_n_1 = AssembleInternalForceVector(ListOfElement,disp_n_1,system_size)
        r_n_1 = CalculateResidualStatic(f_int_n_1,F_master)
        

        #prepare next step
        r_n_norm = ResidualNorm(r_n_1)
        r_n = r_n_1
        disp_n = disp_n_1
        K_n = UpdateStiffnessMatrix(ListOfElement,disp_n,ListOfBc)
        ModifyResidual(r_n,ListOfBc)
        

        print(r_n)



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
    return (F_ext-F_int)

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


def solve_explicit(M_master,K_master,C_master,F_master,Bc_List,d_t,t_end):
    # initialize
    M_master_inv = explicit.InverseLumpedMatrix(M_master)
    disp_n = explicit.CreateInitialDisplacementVector(Bc_List,K_master.shape[0])
    vel_n  = np.zeros((K_master.shape[0],1))
    n, t_n = 0, 0.00

    f_int_n = truss.CalculateInternalForces(K_master,disp_n)
    res_n = explicit.CalculateResidualExplicit(F_master,f_int_n)
    acc_n = explicit.ComputeAcceleration(M_master_inv,C_master,vel_n,res_n)

    # prepare data
    disp_expl = []
    time_expl = []

    # loop over time
    while t_n < t_end:
        
        # half time steps
        t_n_05, t_n_1 = explicit.UpdateTime(t_n,d_t)
        v_n_05 = explicit.UpdateVelocity(vel_n,acc_n,d_t)
        explicit.EnforceBoundaryConditionsVelocity(Bc_List,v_n_05)
        # update displacements + residual
        disp_n_1 = explicit.UpdateDisplacement(v_n_05,disp_n,d_t)
        f_int_n_1 = truss.CalculateInternalForces(K_master,disp_n_1)   #----------> linear part!!!
        res_n_1 = explicit.CalculateResidualExplicit(F_master,f_int_n_1)
        # update acc + vel
        acc_n_1 = explicit.ComputeAcceleration(M_master_inv,C_master,v_n_05,res_n_1)
        vel_n_1 = explicit.UpdateVelocity(v_n_05,acc_n_1,d_t)

        # save date
        time_expl.append(t_n)
        disp_expl.append(disp_n)

        # prepare next time step
        disp_n = disp_n_1
        vel_n = vel_n_1
        acc_n = acc_n_1
        t_n += d_t

    return disp_expl, time_expl


