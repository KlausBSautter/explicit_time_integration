import numpy as np 
import copy
import timeit
import explicit_functions as explicit
import non_linear_solver_functions as nl_solving
import truss_element_linear_functions as truss
import truss_element_non_linear_functions as truss_nl

################################################################################
##################### STATIC SOLVERS ###########################################
################################################################################

def solve_linear(LHS,RHS):
    u = np.linalg.solve(LHS,RHS)
    return u

#ListOfElement [[element1,E,A,nodes]]
def solve_nonlinear_nr_lc(K_T,ListOfElement,ListOfBc,F_master):
    
    ## load control: given load -> solve for displacement
    print('\n' + '################################################# \n')
    print('Starting Newton-Raphson Iteration: ')
    start_time = timeit.default_timer()
    e_tollerance = 10**(-6)
    system_size = K_T.shape[0]
    K_n = copy.deepcopy(K_T)
    disp_n = explicit.CreateInitialDisplacementVector(ListOfBc,system_size)

    f_int_n = nl_solving.AssembleInternalForceVector(ListOfElement,disp_n,system_size)
    r_n = nl_solving.CalculateResidualStatic(f_int_n,F_master)
    nl_solving.ModifyResidual(r_n,ListOfBc)
    r_n_norm = nl_solving.ResidualNorm(r_n)
    n = 0

    while (r_n_norm > e_tollerance):
        n += 1
        disp_n_1 = solve_linear(K_n,r_n)
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

    elapsed = timeit.default_timer() - start_time
    print('\n'+'Finished in: ', elapsed, ' seconds\n')
    print('################################################# \n\n')
    return disp_n

def solve_nonlinear_nr_dc(ListOfElement,ListOfBc,F_master):
    
    ## displacement control: given displacement -> solve for load

    print('\n' + '################################################# \n')
    print('Starting Newton-Raphson Iteration: ')
    start_time = timeit.default_timer()
    e_tollerance = 10**(-6)
    system_size = F_master.shape[0]

    disp_const = explicit.CreateInitialDisplacementVector(ListOfBc,system_size)
    f_ext_0 = nl_solving.CreateInitialForceVector(F_master,ListOfBc)
    lambda_n = np.zeros((system_size,1))
    f_ext_n = nl_solving.MultiplyVectorEntries(f_ext_0,lambda_n)

    f_int_0 = nl_solving.AssembleInternalForceVector(ListOfElement,disp_const,system_size)
    r_n = nl_solving.CalculateResidualStatic(f_int_0,f_ext_n)
    nl_solving.ModifyResidual(r_n,ListOfBc)
    r_n_norm = nl_solving.ResidualNorm(r_n)
    n = 0

    while (r_n_norm > e_tollerance):
        n += 1
        lambda_n_1 = nl_solving.DivideVectorEntries(r_n,-f_ext_0)
        lambda_n_1 = lambda_n - lambda_n_1

        f_ext_n_1 = nl_solving.MultiplyVectorEntries(lambda_n_1,f_ext_0)
        r_n_1 = nl_solving.CalculateResidualStatic(f_int_0,f_ext_n_1)

        #prepare next step
        nl_solving.ModifyResidual(r_n_1,ListOfBc)
        r_n_norm = nl_solving.ResidualNorm(r_n_1)
        r_n = r_n_1
        lambda_n = lambda_n_1
        nl_solving.PrintSolverUpdate(r_n,r_n_norm,n)

    elapsed = timeit.default_timer() - start_time
    print('\n'+'Finished in: ', elapsed, ' seconds\n')
    print('################################################# \n\n')
    return lambda_n

################################################################################
##################### DYNAMIC SOLVERS ###########################################
################################################################################

def solve_explicit_linear(M_master,K_master,C_master,F_master,Bc_List,d_t,t_end):
    
    print('\n' + '################################################# \n')
    print('Starting Explicit Time Integration: ')
    start_time = timeit.default_timer()

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

    elapsed = timeit.default_timer() - start_time
    print('\n'+'Finished in: ', elapsed, ' seconds\n')
    print('################################################# \n\n')

    return disp_expl, time_expl


def solve_explicit_non_linear(M_master,K_master,C_master,F_master,F_mod,ListOfElements,Bc_List,d_t,t_end):
    # initialize
    M_master_inv = explicit.InverseLumpedMatrix(M_master)
    disp_n = explicit.CreateInitialDisplacementVector(Bc_List,K_master.shape[0])
    vel_n  = np.zeros((K_master.shape[0],1))
    n, t_n = 0, 0.00



    print('\n' + '################################################# \n')
    print('Starting Explicit Time Integration: ')
    start_time = timeit.default_timer()
    f_int_n = solve_nonlinear_nr_dc(ListOfElements,Bc_List,F_mod)



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


        bc_list_n_1 = explicit.UpdateNonLinearDisplacementVector(disp_n_1)
        f_int_n_1 = solve_nonlinear_nr_dc(ListOfElements,bc_list_n_1,F_mod)

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

    elapsed = timeit.default_timer() - start_time
    print('\n'+'Finished in: ', elapsed, ' seconds\n')
    print('################################################# \n\n')

    return disp_expl, time_expl