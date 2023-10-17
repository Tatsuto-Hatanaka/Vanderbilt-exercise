import numpy as np
import matplotlib.pyplot as plt
from pythtb import *

def setmodel(t, del_t, Delta):
    lat = [[1.0]]
    orb = [[0],[0.5]]
    my_model = tb_model(1,1,lat,orb)
    
    my_model.set_onsite([Delta,-Delta])
    my_model.set_hop(t+del_t, 0, 1, [0])
    my_model.set_hop(t-del_t, 1, 0, [1])
    return my_model

def calc_BP1(model, nk=61):
    evec_array = wf_array(model,[nk])       # set array dimension
    evec_array.solve_on_grid([0.])           # fill with eigensolutions
    berry_phase = evec_array.berry_phase([0])  # Berry phase of bottom band
    print("Berry phase is %7.3f pi"% (berry_phase/np.pi))
    
def calc_BP2(model, nk=61):
    (k_vec,k_dist,k_node) = model.k_path('full',nk,report=False)
    (eval,evec) = model.solve_all(k_vec, eig_vectors=True)
    evec=evec[0]   # pick band=0 from evec[band,kpoint,orbital]
                   # now just evec[kpoint,orbital]
    # k-points 0 and 60 refer to the same point on the unit circle
    # so we will work only with evec[0],...,evec[59]
    # compute Berry phase of lowest band
    prod = 1.+0.j
    for i in range(1,nk-1):
        prod *= np.vdot(evec[i-1],evec[i])
    phase = np.exp((-2.j)*np.pi*model.get_orb().reshape(-1))
    evec_last = phase*evec[0]
    prod *= np.vdot(evec[-2],evec_last)  # include <evec_59|evec_last>
    print("Berry phase is %7.3f pi"% (-np.angle(prod)/np.pi))

# inversion symmetry about a site (del_t = 0)
model1 = setmodel(1.0, 0.0, 0.4)
# inversion symmetry about a bond (Delta = 0)
model2 = setmodel(1.0, 0.2, 0.0)
# no inversion symmetry (del_t != 0, Delta != 0)
model3 = setmodel(1.0, 0.2, 0.2)

for i,model in enumerate([model1,model2,model3]):
    print(i+1)
    calc_BP1(model)
    calc_BP2(model)