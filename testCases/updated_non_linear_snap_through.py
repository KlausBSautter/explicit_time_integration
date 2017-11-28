from import_custom_modules import *


################################################################################
#################   POSTPROCESSING #############################################
################################################################################
E, A = 2.1*(10**11), 0.01
F = 0

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

F_end = -3.5*(10**7)
dF = -200000
U_updated = explicit.CreateInitialDisplacementVector(Bc_List,F_master.shape[0])

disp_nr = []
disp_up = []
force = []
force_up = []

while (abs(F) < abs(F_end)):
    U_non_linear_static = solver.solve_nonlinear_nr_lc(K_mod,Element_List,Bc_List,F_mod)

    force.append(abs(F))
    disp_nr.append(abs(U_non_linear_static.T[0,3]))
    
    F += dF
    F_master = np.matrix([[0.00,0.00,0.00,F]]).T   
    FFFF, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)    
    print('-------------------------------------------------------------------->    ', F/F_end*100, ' %\n')
    #print('non_linear disp: ', U_non_linear_static.T[0,3])


F = 0
dF = -2000000

F_master = np.matrix([[0.00,0.00,0.00,F]]).T
FFFF, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)

while (abs(F) < abs(F_end)):
    U_updated += solver.solve_nonlinear_updated(Element_List,Bc_List,F_mod,U_updated)
    force_up.append(abs(F))
    disp_up.append(abs(U_updated.T[0,3]))

    F += dF
    F_master = np.matrix([[0.00,0.00,0.00,dF]]).T   
    FFFF, F_mod = truss.ModifyMasterMatrix(K_master,F_master,Bc_List)



plt.plot(disp_up,force_up,label ='updated')
plt.plot(disp_nr,force,'--',label ='NR')
plt.legend()
plt.grid()
plt.show()


