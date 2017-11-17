import numpy as np 
import copy
import explicit_functions as explicit
import non_linear_solver_functions as nl_solving
import truss_element_linear_functions as truss
import truss_element_non_linear_functions as truss_nl



def solve_linear(LHS,RHS):
    u = np.linalg.solve(LHS,RHS)
    return u

#ListOfElement [[element1,E,A,nodes]]
def solve_nonlinear_nr(K_T,ListOfElement,ListOfBc,F_master):

    print('Starting Newton-Raphson Iteration: ')
    e_tollerance = 10**(-6)
    system_size = K_T.shape[0]
    K_n = copy.deepcopy(K_T)
    disp_n = explicit.CreateInitialDisplacementVector(ListOfBc,system_size)

    f_int_n = nl_solving.AssembleInternalForceVector(ListOfElement,disp_n,system_size)
    r_n = nl_solving.CalculateResidualStatic(f_int_n,F_master)
    r_n_norm = nl_solving.ResidualNorm(r_n)
    n = 0
    while (r_n_norm > e_tollerance):
        n += 1
        disp_n_1 = np.linalg.solve(K_n,r_n)
        disp_n_1 = disp_n - disp_n_1

        f_int_n_1 = nl_solving.AssembleInternalForceVector(ListOfElement,disp_n_1,system_size)
        r_n_1 = nl_solving.CalculateResidualStatic(f_int_n_1,F_master)

        #prepare next step
        nl_solving.ModifyResidual(r_n_1,ListOfBc)
        r_n_norm = nl_solving.ResidualNorm(r_n_1)
        r_n = r_n_1
        disp_n = disp_n_1
        K_n = nl_solving.UpdateStiffnessMatrix(ListOfElement,disp_n,ListOfBc)

        nl_solving.PrintSolverUpdate(r_n,r_n_norm,n)

    return disp_n


def solve_explicit_linear(M_master,K_master,C_master,F_master,Bc_List,d_t,t_end):
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

        # save data
        time_expl.append(t_n)
        disp_expl.append(disp_n)

        # prepare next time step
        disp_n = disp_n_1
        vel_n = vel_n_1
        acc_n = acc_n_1
        t_n += d_t

    return disp_expl, time_expl


