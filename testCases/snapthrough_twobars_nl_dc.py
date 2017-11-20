from import_custom_modules import *


################################################################################
#################   POSTPROCESSING #############################################
################################################################################
E, A, rho = 2.1*(10**11), 0.01, 7850
F = -2.4*(10**7)

# nodes
nodes = [[1,0.00,0.00],[2,2.00,1.00],[3,4.00,0.00]]

# neumann bc.
F_master = np.matrix([[0.00,0.00,0.00,0.0,0.00,0.00]]).T

# dirichlet bc. (#dofs,#disp)
Bc_List = [[0,0.0],[1,0.0],[2,0.00],[3,-0.16751849],[4,0.0],[5,0.00]]

# element stiffness matrices (E,A,L,alpha)
Element1_K = truss_nl.ElementStiffMatrix(E,A,[nodes[0],nodes[1]],explicit.CreateInitialDisplacementVector(Bc_List,F_master.shape[0]))
Element2_K = truss_nl.ElementStiffMatrix(E,A,[nodes[1],nodes[2]],explicit.CreateInitialDisplacementVector(Bc_List,F_master.shape[0]))



# assemble master matrices (#elements,#nodes)
Element_List_K = [[Element1_K,E,A,[nodes[0],nodes[1]]],[Element2_K,E,A,[nodes[1],nodes[2]]]]
K_master = truss.AssembleElementMatrices(Element_List_K)

# modify matrices and vector
K_mod, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)


################################################################################
#################   SOLVING        #############################################
################################################################################

#### solve non linear static
F_non_linear_static = solver.solve_nonlinear_nr_dc(Element_List_K,Bc_List,F_mod)

print('############ RESULTS ############')
print('non_linear Force: ', F_non_linear_static.T)




