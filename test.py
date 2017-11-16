import numpy as np
import truss_element_linear_functions as truss
import solver_functions as solver
import explicit_functions as explicit
import general_functions as general


################################################################################
#################   POSTPROCESSING #############################################
################################################################################

# element stiffness matrices (E,A,L,alpha)
Element1_K = truss.ElementStiffMatrix(2.1*(10**11),0.01,0.50,0.00)
Element2_K = truss.ElementStiffMatrix(2.1*(10**11),0.01,0.50,0.00)

# element mass matrics (rho,A,L)
Element1_M = truss.ElementMassMatrix(7850,0.01,0.50)
Element2_M = truss.ElementMassMatrix(7850,0.01,0.50)

# assemble master matrices (#elements,#nodes)
K_master = truss.AssembleElementMatrices([Element1_K,Element2_K],[[1,2],[2,3]])
M_master = truss.AssembleElementMatrices([Element1_M,Element2_M],[[1,2],[2,3]])

# neumann bc.
F_master = np.matrix([[0.00,0.00,0.00,0.00,100000.00,0.00]]).T

# dirichlet bc. (#dofs,#disp)
Bc_List = [[0,0.0],[1,0.0],[3,0.00],[5,0.0]]

# modify matrices and vector
K_mod, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)


################################################################################
#################   SOLVING        #############################################
################################################################################

##### solve linear static
U_linear_static = solver.solve_linear(K_mod,F_mod)
F_linear_static = np.dot(K_master,U_linear_static)

#### solve linear dynamic explicit (no damping yet)
# initialize
M_master_inv = explicit.InverseLumpedMatrix(M_master)
disp_n = explicit.CreateInitialDisplacementVector(Bc_List,K_master.shape[0])
vel_n  = np.zeros((K_master.shape[0],1))
n, t_n, d_t, t_end = 0, 0.00, 0.00001, 0.004

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



general.PrintDisplacement(disp_expl,time_expl,[4,2],'Explicit Time Integration')



