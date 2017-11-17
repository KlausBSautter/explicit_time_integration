from import_custom_modules import *


################################################################################
#################   POSTPROCESSING #############################################
################################################################################
E, A = 2.1*(10**11), 0.01

# nodes
nodes = [[1,0.00,0.00],[2,0.50,0.00],[3,1.00,0.00]]

# neumann bc.
F_master = np.matrix([[0.00,0.00,0.00,0.00,0.00,0.00]]).T

# dirichlet bc. (#dofs,#disp)
Bc_List = [[0,0.0],[1,0.0],[3,0.00],[5,0.00],[4,0.10],[2,-0.10]]

# element stiffness matrices (E,A,L,alpha)
Element1_K = truss.ElementStiffMatrix(E,A,[nodes[0],nodes[1]])
Element2_K = truss.ElementStiffMatrix(E,A,[nodes[1],nodes[2]])

# element mass matrics (rho,A,L)
Element1_M = truss.ElementMassMatrix(7850,A,[nodes[0],nodes[1]])
Element2_M = truss.ElementMassMatrix(7850,A,[nodes[1],nodes[2]])

# assemble master matrices (#elements,#nodes)
Element_List_K = [[Element1_K,E,A,[nodes[0],nodes[1]]],[Element2_K,E,A,[nodes[1],nodes[2]]]]
Element_List_M = [[Element1_M,E,A,[nodes[0],nodes[1]]],[Element2_M,E,A,[nodes[1],nodes[2]]]]

K_master = truss.AssembleElementMatrices(Element_List_K)
M_master = truss.AssembleElementMatrices(Element_List_M)


# modify matrices and vector
K_mod, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)
M_mod, FFFF = truss.ModifyMasterMatrix(M_master,F_master,Bc_List)

# create DampingMatrix
C_master = np.zeros((K_master.shape[0],K_master.shape[0]))
#C_master = truss.MasterDampingMatrix(K_mod,M_mod,Bc_List,0.05)



################################################################################
#################   SOLVING        #############################################
################################################################################

##### solve linear static
U_linear_static = solver.solve_linear(K_mod,F_mod)
F_linear_static = np.dot(K_master,U_linear_static)

#### solve linear dynamic explicit (no damping yet)
disp_expl, time_expl =  solver.solve_explicit_linear(M_master,K_master,C_master,F_master,Bc_List,0.00001, 0.004)

general.PrintDisplacement(disp_expl,time_expl,[2,4],'Explicit Time Integration')



