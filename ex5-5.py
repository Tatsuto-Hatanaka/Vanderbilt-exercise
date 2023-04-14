from pythtb import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

def set_model(b,t):
    lat = [[1,0],[0,1]]
    orb = [[0,0]]
    sigma_x=np.array([0.,1.,0.,0])
    sigma_y=np.array([0.,0.,1.,0])
    sigma_z=np.array([0.,0.,0.,1])
    model = tb_model(2,2,lat,orb,nspin=2)
    model.set_onsite([b*sigma_z])
    model.set_hop(t*(sigma_z+1j*sigma_x),0,0,[1,0])
    model.set_hop(t*(sigma_z+1j*sigma_y),0,0,[0,1])
    return model

def calcu_chern(model,nk):
    dk = 2.*np.pi/(nk-1)
    my_array = wf_array(model,[nk,nk])
    my_array.solve_on_grid([0.,0.])
    # bcurv = my_array.berry_flux([0],individual_phases=True)/(dk*dk)
    chern = my_array.berry_flux([0])/(2.*np.pi)
    return chern

bmin, bmax, nb = 0, 5, 21
tmin, tmax, nt = 0, 1, 21
bs = np.linspace(bmin,bmax,nb)
ts = np.linspace(tmin,tmax,nt)
b_ = []
t_ = []
chern_ = []
fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.set_xlim(bmin,bmax)
ax.set_ylim(tmin,tmax)
ax.set_xlabel("$b$")
ax.set_ylabel("$t$")
ax.set_title("Phase diagram")

for ib, b in enumerate(bs):
    for it, t in enumerate(ts):
        model = set_model(b,t)
        chern = calcu_chern(model,61)
        chern_.append(chern.round(2))
        b_.append(b)
        t_.append(t)
sc = ax.scatter(b_, t_, c=chern_)
legend = ax.legend(sc.legend_elements()[0], ["-1","0"], loc='best', title='Chern')
ax.add_artist(legend)
fig.savefig("figure/ex5-5.pdf")