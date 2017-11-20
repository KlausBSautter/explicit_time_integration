import copy
import numpy as np




def InverseLumpedMatrix(LumpedMatrix):
    InverseMatrix = copy.deepcopy(LumpedMatrix)
    matrix_size = InverseMatrix.shape[0]
    for i in range(matrix_size):
        InverseMatrix[i,i] = 1.00 / InverseMatrix[i,i]
    return InverseMatrix


def CreateInitialDisplacementVector(ListOfBc,number_dof):
    d_0 = np.zeros((number_dof,1))
    number_bc = len(ListOfBc)

    for i in range(number_bc):
        current_dof = ListOfBc[i][0]
        current_disp = ListOfBc[i][1]

        d_0[current_dof,0] = current_disp
    return d_0


def UpdateTime(t_n,d_t):
    t_n_1 = t_n + d_t
    t_n_05 = 0.50 * (t_n + t_n_1)
    return t_n_05, t_n_1 


def ComputeAcceleration(invM,C,vel,res):
    b = res - np.dot(C,vel)
    a = np.dot(invM,b)
    return a

def UpdateVelocity(vel,acc,d_t):
    return (vel + (acc * d_t * 0.50))



def CalculateResidualExplicit(f_ext,f_int):
    return (f_ext-f_int)


def EnforceBoundaryConditionsVelocity(ListOfBc,vel):
    numerical_limit = 10.0**(-16)
    number_bc = len(ListOfBc)

    for i in range(number_bc):
        current_dof = ListOfBc[i][0]
        current_disp = ListOfBc[i][1]

        if (abs(current_disp)<numerical_limit): vel[current_dof] = 0.00


def UpdateDisplacement(vel,disp,d_t):
    return (disp + (vel * d_t))


def UpdateNonLinearDisplacementVector(disp):
    system_size = disp.shape[0]
    bc_list_return = []
    for i in range(system_size):
        bc_list_return.append([i,disp[i,0]])
    return bc_list_return