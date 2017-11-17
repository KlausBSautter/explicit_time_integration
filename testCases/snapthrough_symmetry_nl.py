from import_custom_modules import *


################################################################################
#################   POSTPROCESSING #############################################
################################################################################
E, A = 2.1*(10**11), 0.01
F = -2.4*(10**7)

# nodes
nodes = [[1,0.00,0.00],[2,2.00,1.00]]

# neumann bc.
F_master = np.matrix([[0.00,0.00,0.00,F]]).T

# dirichlet bc. (#dofs,#disp)
Bc_List = [[0,0.0],[1,0.0],[2,0.00]]

# element stiffness matrices (E,A,L,alpha)
Element1_K = truss_nl.ElementStiffMatrix(E,A,[nodes[0],nodes[1]],explicit.CreateInitialDisplacementVector(Bc_List,F_master.shape[0]))

# assemble master matrices (#elements,#nodes)
Element_List = [[Element1_K,E,A,[nodes[0],nodes[1]]]]
K_master = truss.AssembleElementMatrices(Element_List)



# modify matrices and vector
K_mod, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)



################################################################################
#################   SOLVING        #############################################
################################################################################

##### solve linear static
U_linear_static = solver.solve_linear(K_mod,F_mod)
F_linear_static = np.dot(K_master,U_linear_static)



#### solve non linear static
U_non_linear_static = solver.solve_nonlinear_nr(K_mod,Element_List,Bc_List,F_mod)

print('############ RESULTS ############')
print('linear disp: ', U_linear_static.T)
print('non_linear disp: ', U_non_linear_static.T)

#### solve linear dynamic explicit (no damping yet)
#disp_expl, time_expl =  solver.solve_explicit(M_master,K_master,C_master,F_master,Bc_List,0.00001, 0.004)
#general.PrintDisplacement(disp_expl,time_expl,[2,4],'Explicit Time Integration')



