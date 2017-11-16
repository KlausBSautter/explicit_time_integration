import matplotlib.pyplot as plt

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
    plt.title(Head)
    plt.show()