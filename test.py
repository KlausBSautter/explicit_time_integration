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
disp_expl, time_expl =  solver.solve_explicit(M_master,K_master,F_master,Bc_List,0.00001, 0.004)



general.PrintDisplacement(disp_expl,time_expl,[4,2],'Explicit Time Integration')



