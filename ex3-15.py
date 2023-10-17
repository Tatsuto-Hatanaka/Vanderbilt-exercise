import numpy as np
import matplotlib.pyplot as plt
from pythtb import *

lat = [[1.0]]
orb = [[0],[0.5]]
my_model = tb_model(1,1,lat,orb)

def setloop(model,t=1.0,nkp=61,N=51,r=0.1,c=[0.0,0.0]):
    twopi = 2*np.pi
    thetas = np.linspace(0,twopi,N)
    Deltas = r*np.cos(thetas)+c[0]
    del_ts = r*np.sin(thetas)+c[1]
    berry_phases = np.zeros(N)
    for i in range(N):
        Delta = Deltas[i]
        del_t = del_ts[i]
        model.set_onsite([Delta,-Delta],mode="reset")
        model.set_hop(t+del_t, 0, 1, [0],mode="reset")
        model.set_hop(t-del_t, 1, 0, [1],mode="reset")
        evec_array = wf_array(model,[nkp])       # set array dimension
        evec_array.solve_on_grid([0.])           # fill with eigensolutions
        berry_phases[i] = evec_array.berry_phase([0])
    return thetas, Deltas, del_ts, berry_phases

# C1 : x^2+y^2=(0.1)^2
thetas, deltas1, del_ts1, bps1 = setloop(my_model)
# C2 : x^2+y^2=(0.2)^2
thetas, deltas2, del_ts2, bps2 = setloop(my_model,r=0.2)
# C3 : (x-0.1)^2+(y-0.1)^2=(0.1)^2
thetas, deltas3, del_ts3, bps3 = setloop(my_model,c=[0.1,0.1])
# C4 : (x-0.05)^2+(y-0.15)^2=(0.12)^2
thetas, deltas4, del_ts4, bps4 = setloop(my_model,r=0.12,c=[-0.05,-0.15])
# C5 : (x-0.1)^2+(y-0.1)^2=(0.16)^2
thetas, deltas5, del_ts5, bps5 = setloop(my_model,r=0.16,c=[0.1,0.1])

twopi = 2*np.pi
N=51
thetas = np.linspace(0,twopi,N)

fig, ax = plt.subplots(2,5, figsize=(14,6))

ax[0,0].scatter(deltas1, del_ts1, s=2)
ax[0,1].scatter(deltas2, del_ts2, s=2)
ax[0,2].scatter(deltas3, del_ts3, s=2)
ax[0,3].scatter(deltas4, del_ts4, s=2)
ax[0,4].scatter(deltas5, del_ts5, s=2)
ax[1,0].plot(thetas, bps1, "-o", ms=2)
ax[1,1].plot(thetas, bps2, "-o", ms=2)
ax[1,2].plot(thetas, bps3, "-o", ms=2)
ax[1,3].plot(thetas, bps4, "-o", ms=2)
ax[1,4].plot(thetas, bps5, "-o", ms=2)
for i in range(5):
    if i==0:
        ax[0,i].set_xlabel("$\Delta$")
        ax[0,i].set_ylabel("$\delta$")
        ax[1,i].set_xlabel("θ")
        ax[1,i].set_ylabel("Berry Phase")
    ax[0,i].scatter([0.],[0.])
    ax[0,i].set_xlim(-0.3,0.3)
    ax[0,i].set_ylim(-0.3,0.3)
    ax[0,i].grid()
    ax[1,i].set_xlim(0,twopi)
    ax[1,i].set_ylim(-twopi,twopi)

fig.savefig("figure/ex3-15.pdf")