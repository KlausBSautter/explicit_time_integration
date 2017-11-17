import numpy as np

E, A = 2.1*(10.0**11.0), 0.01
L = np.sqrt(4.0+1.0)
F = 2.4*(10**7)



def residual(v):
    a = (E*A)/(2*L*L*L)
    b = (v**3) - 3*(v**2) + 2*v
    return((a*b)-F)

def K(v):
    a = (E*A)/(2*L*L*L)
    b = 3*(v**2) - 6*v + 2
    return(a*b)





v_i = 0.00+0.127775313+0.0365690172423+0.00315112214599+2.30374478194e-05

print('r: ',residual(v_i))
print('K: ',K(v_i))
print('r/K: ', residual(v_i)/K(v_i))


print(v_i)