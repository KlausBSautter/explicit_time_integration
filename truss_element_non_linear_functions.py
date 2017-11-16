import numpy as np



def ElementStiffMatrix(E,A,L,alpha):
    c = np.cos(alpha)
    s = np.sin(alpha)
    K = np.matrix([[c*c,s*c,-c*c,-s*c],
                   [s*c,s*s,-s*c,-s*s],
                   [-c*c,-s*c,c*c,s*c],
                   [-s*c,-s*s,s*c,s*s]])
    K = K * (E*A/L)
    return K



def CalculateCurrentLength(L,disp,nodes):
    nodeA, nodeB = nodes[0], nodes[1]
    ui,uj = disp[(nodeA-1)*2], disp[(nodeB-1)*2] 
    vi,vj = disp[((nodeA-1)*2)+1], disp[((nodeB-1)*2)+1] 

