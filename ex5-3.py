from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

Delta = 5.0
t_0 = 1.0
tprime = 0.4

def set_model(Delta,t_0,tprime):
    # set geometry
    lat=[[1.0,0.0],[0.0,1.0]]
    orb=[[0.0,0.0],[0.5,0.5]]
    my_model=tb_model(2,2,lat,orb)
    # set model
    my_model.set_onsite([-Delta,Delta])
    my_model.set_hop(-t_0, 0, 0, [ 1, 0])
    my_model.set_hop(-t_0, 0, 0, [ 0, 1])
    my_model.set_hop( t_0, 1, 1, [ 1, 0])
    my_model.set_hop( t_0, 1, 1, [ 0, 1])
    my_model.set_hop( tprime , 1, 0, [ 1, 1])
    my_model.set_hop( tprime*1j, 1, 0, [ 0, 1])
    my_model.set_hop(-tprime , 1, 0, [ 0, 0])
    my_model.set_hop(-tprime*1j, 1, 0, [ 1, 0])
    # my_model.display()
    return my_model

def calcu_band(my_model,path):
    # generate k-point path and labels and solve Hamiltonian
    (k_vec,k_dist,k_node) = my_model.k_path(path,121,report=False)
    evals = my_model.solve_all(k_vec)
    return k_vec, k_dist, k_node, evals

def calcu_chern(my_model,nk):
    dk = 2.*np.pi/(nk-1)
    my_array = wf_array(my_model,[nk,nk])
    my_array.solve_on_grid([0.,0.])
    # bcurv = my_array.berry_flux([0],individual_phases=True)/(dk*dk)
    chern = my_array.berry_flux([0])/(2.*np.pi)
    return chern

path=[[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
k_lab=(r'$\Gamma $',r'$X$', r'$M$', r'$\Gamma $')

colors = ['#FF0000', '#FF3F00', '#FF7F00', '#FFBF00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#2E2B5F', '#8B00FF', '#FF00FF']
cmap = LinearSegmentedColormap.from_list('mycmap', colors)

fig, ax = plt.subplots(2,2, figsize=(10.,6.))


#---------------b1-----------------
deltas = np.linspace(0,10,11)
ax[0,0].set_title("(b1) $t_0={0}, t'={1}$".format(1.0,0.0))
for i in range(len(deltas)):
    my_model = set_model(Delta=deltas[i],t_0=1.0,tprime=0.)
    k_vec, k_dist, k_node, evals = calcu_band(my_model,path)
    if i==0:
        ax[0,0].set_xlim([0,k_node[-1]])
        ax[0,0].set_xticks(k_node)
        ax[0,0].set_xticklabels(k_lab)
        for n in range(len(k_node)):
            ax[0,0].axvline(x=k_node[n], linewidth=0.5, color='k')
    ax[0,0].plot(k_dist,evals[0],color=cmap(i/(len(colors)-1)),label="$\Delta={}$".format(deltas[i]))
    ax[0,0].plot(k_dist,evals[1],color=cmap(i/(len(colors)-1)))
ax[0,0].legend()
# ax.legend(loc="upper left", bbox_to_anchor=(1, 1),mode="expand")


#---------------b2-----------------
tprimes = np.linspace(0,0.4,11)
ax[0,1].set_title("(b2) $\Delta={0}, t_0={1}$".format(3.0,1.0))
for i in range(len(tprimes)):
    my_model = set_model(Delta=3.0,t_0=1.0,tprime=tprimes[i])
    k_vec, k_dist, k_node, evals = calcu_band(my_model,path)
    if i==0:
        ax[0,1].set_xlim([0,k_node[-1]])
        ax[0,1].set_xticks(k_node)
        ax[0,1].set_xticklabels(k_lab)
        for n in range(len(k_node)):
            ax[0,0].axvline(x=k_node[n], linewidth=0.5, color='k')
    ax[0,1].plot(k_dist,evals[0],color=cmap(i/(len(colors)-1)),label="$t'={}$".format(tprimes[i]))
    ax[0,1].plot(k_dist,evals[1],color=cmap(i/(len(colors)-1)))
# ax[0,1].set_ylim(-2.5,2.5)
ax[0,1].legend()

#----------------b3-----------------
t0s = np.linspace(0.5,1.5,11)
ax[1,0].set_title("(c) $\Delta={0}, t'={1}$".format(5.0,0.4))
for it, t_0 in enumerate(t0s):
    my_model = set_model(Delta=5.0,t_0=t_0,tprime=0.4)
    k_vec, k_dist, k_node, evals = calcu_band(my_model,path)
    if it==0:
        ax[1,0].set_xlim([0,k_node[-1]])
        ax[1,0].set_xticks(k_node)
        ax[1,0].set_xticklabels(k_lab)
        for n in range(len(k_node)):
            ax[0,0].axvline(x=k_node[n], linewidth=0.5, color='k')
    ax[1,0].plot(k_dist,evals[0],color=cmap(it/(len(colors)-1)),label="$t_0={}$".format(t_0.round(2)))
    ax[1,0].plot(k_dist,evals[1],color=cmap(it/(len(colors)-1)))
# ax[1,0].set_ylim(-2.5,2.5)
ax[1,0].legend()

fig.savefig("figure/ex5-3b.pdf")

#----------------c-----------------
nk  = 31
t_0 = 1.0
delmin, delmax = 0, 5
deltas         = np.linspace(delmin,delmax,11)
t0min, t0max   = 0.5, 1.5
t0s            = np.linspace(t0min,t0max,11)
tpmin, tpmax   = 0, 1.0
tprimes = np.linspace(tpmin,tpmax,11)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.set_title("(c) Chern number".format(1.0))
ax.set_xlim(delmin,delmax)
ax.set_ylim(t0min,t0max)
ax.set_zlim(tpmin,tpmax)
ax.set_xlabel("$\Delta$")
ax.set_ylabel("$t_0$")
ax.set_zlabel("$t'$")
for id,delta in enumerate(deltas):
    for it0, t_0 in enumerate(t0s):
        for itp,tprime in enumerate(tprimes):
            my_model = set_model(Delta=delta,t_0=t_0,tprime=tprime)
            chern = calcu_chern(my_model,nk)
            print(delta, t_0, tprime, chern)
            # c = "k" if np.isclose(chern,0) else "r"
            # ax.scatter(delta,t_0,tprime,color=c)
            if np.isclose(chern,0): continue
            else: ax.scatter(delta,t_0,tprime,color="r")

ax.view_init(elev=20,azim=45)
fig.savefig("figure/ex5-3c.pdf")