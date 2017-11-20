import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def PrintDofDisplacement(disp,time,dof):
    len_array = len(time)
    dof_disp = []
    for i in range(len_array):
        dof_disp.append(disp[i][dof])
    plt.plot(time,dof_disp, label='dof ' + str(dof))

    

def PrintDisplacement(disp,time,ListOfDof,Head):
    number_dof = len(ListOfDof)
    for i in range(number_dof):
        PrintDofDisplacement(disp,time,ListOfDof[i])
    plt.legend()
    plt.grid(True)
    plt.xlabel('time t')
    plt.ylabel('displacement')
    plt.title(Head)

def ShowPrint():
    plt.show()



def FindEigenValues(MassMatrix,StiffnessMatrix,ListOfBc):
    number_of_Bc = len(ListOfBc)
    old_mat_size = MassMatrix.shape[0]
    new_mat_size = old_mat_size - number_of_Bc

    M_mod = sp.zeros(old_mat_size,old_mat_size)
    K_mod = sp.zeros(old_mat_size,old_mat_size)

    for i in range(old_mat_size):
        for j in range(old_mat_size):
            M_mod[i,j] = MassMatrix[i,j]
            K_mod[i,j] = StiffnessMatrix[i,j]

    alpha = sp.Symbol('alpha')

    MK = K_mod - M_mod * alpha
    MK_det = MK.det()
    eigenvalues = sp.solve(MK_det,alpha)
    eigen_array = []
    for i in range(new_mat_size): eigen_array.append(np.sqrt(np.float(eigenvalues[i])))
    return eigen_array