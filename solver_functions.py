import numpy as np 
import explicit_functions as explicit
import truss_element_linear_functions as truss



def solve_linear(LHS,RHS):
    u = np.linalg.solve(LHS,RHS)
    return u



def solve_explicit(M_master,K_master,F_master,Bc_List,d_t,t_end):
    # initialize
    M_master_inv = explicit.InverseLumpedMatrix(M_master)
    disp_n = explicit.CreateInitialDisplacementVector(Bc_List,K_master.shape[0])
    vel_n  = np.zeros((K_master.shape[0],1))
    n, t_n = 0, 0.00

    f_int_n = truss.CalculateInternalForces(K_master,disp_n)
    res_n = explicit.CalculateResidualExplicit(F_master,f_int_n)
    acc_n = explicit.ComputeAcceleration(M_master_inv,res_n)

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
        acc_n_1 = explicit.ComputeAcceleration(M_master_inv,res_n_1)
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


