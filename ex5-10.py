from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def set_checkerboard_model(Delta,t_0,tprime,nspin=1,chern=-1,rashba=None):
    lat=[[1.0,0.0],[0.0,1.0]]
    orb=[[0.0,0.0],[0.5,0.5]]
    my_model=tb_model(2,2,lat,orb,nspin=nspin)
    sigma_x=np.array([0.,1.,0.,0])
    sigma_y=np.array([0.,0.,1.,0])
    sigma_z=np.array([0.,0.,0.,1])
    my_model.set_onsite([-Delta,Delta])
    my_model.set_hop(-t_0, 0, 0, [ 1, 0])
    my_model.set_hop(-t_0, 0, 0, [ 0, 1])
    my_model.set_hop( t_0, 1, 1, [ 1, 0])
    my_model.set_hop( t_0, 1, 1, [ 0, 1])
    if nspin==2:
        my_model.set_hop( tprime, 1, 0, [ 1, 1])
        my_model.set_hop( tprime*1j*sigma_z, 1, 0, [ 0, 1])
        my_model.set_hop(-tprime, 1, 0, [ 0, 0])
        my_model.set_hop(-tprime*1j*sigma_z, 1, 0, [ 1, 0])
    elif nspin==1:
        my_model.set_hop( tprime, 1, 0, [ 1, 1])
        my_model.set_hop(-chern*tprime*1j, 1, 0, [ 0, 1])
        my_model.set_hop(-tprime, 1, 0, [ 0, 0])
        my_model.set_hop( chern*tprime*1j, 1, 0, [ 1, 0])
    if rashba:
        my_model.set_hop( 1j*rashba*sigma_x, 0, 0, [ 1, 0], mode="add")
        my_model.set_hop(-1j*rashba*sigma_y, 0, 0, [ 0, 1], mode="add")
    return my_model

def calcu_band(my_model,path):
    (k_vec,k_dist,k_node) = my_model.k_path(path,121,report=False)
    evals = my_model.solve_all(k_vec)
    return k_vec, k_dist, k_node, evals

path=[[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
k_lab=(r'$\Gamma $',r'$X$', r'$M$', r'$\Gamma $')


#-------------------- a ----------------------
Delta = 3.0
t_0 = 1.0
tprime = 0.4
model_trs = set_checkerboard_model(Delta=Delta,t_0=t_0,tprime=tprime,nspin=2)
model_up = set_checkerboard_model(Delta=Delta,t_0=t_0,tprime=tprime,nspin=1,chern=-1)
model_dn = set_checkerboard_model(Delta=Delta,t_0=t_0,tprime=tprime,nspin=1,chern=1)
# model_trs.display()

fig, ax = plt.subplots(1,2, figsize=(10.,4.))
k_vec, k_dist, k_node, evals = calcu_band(model_trs,path)
ax[0].set_title("(a) band structure")
ax[0].set_xlim([0,k_node[-1]])
ax[0].set_xticks(k_node)
ax[0].set_xticklabels(k_lab)
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], linewidth=0.5, color='k')
for ib in range(evals.shape[0]):
    ax[0].plot(k_dist,evals[ib],label="band {}".format(ib))
ax[0].legend()

ax[1].set_title("(a) Centers of hWF")
ax[1].set_xlabel(r"k vector along direction 0")
ax[1].set_ylabel(r"Wannier center along direction 1")
nk0,nk1  = 101,11
k_vec   = np.linspace(0,2*np.pi,nk0)
k_node  = [0.0, np.pi, 2*np.pi]
k_label = [r"$0$",r"$\pi$", r"$2\pi$"]
phi_1   = []
occs    = [[0,1],[0],[0]]
for im, model in enumerate([model_trs,model_up,model_dn]):
    my_array =wf_array(model,[nk0,nk1])
    my_array.solve_on_grid([0.0,0.0])
    phi_1.append(my_array.berry_phase(occ=occs[im], dir=1, contin=True))
ax[1].set_xlim(0,2*np.pi)
ax[1].set_xticks(k_node)
ax[1].set_xticklabels(k_label)
ax[1].set_ylim(-0.5,3+0.5)
for j in range(-1,nk1+1):
    if j==0:
        ax[1].plot(k_vec,float(j)+phi_1[0]/(2.0*np.pi),'k-',zorder=-50,label="TR invariant")
        ax[1].plot(k_vec,float(j)+phi_1[1]/(2.0*np.pi),'r-',zorder=-50,label="spin up")
        ax[1].plot(k_vec,float(j)+phi_1[2]/(2.0*np.pi),'b-',zorder=-50,label="spin dn")
    else:
        ax[1].plot(k_vec,float(j)+phi_1[0]/(2.0*np.pi),'k-',zorder=-50)
        ax[1].plot(k_vec,float(j)+phi_1[1]/(2.0*np.pi),'r-',zorder=-50)
        ax[1].plot(k_vec,float(j)+phi_1[2]/(2.0*np.pi),'b-',zorder=-50)
ax[1].legend()

fig.tight_layout()
fig.savefig("figure/ex5-10a.pdf")



#-------------------- b ----------------------
Delta  = 3.0
t_0    = 1.0
tprime = 0.4
rashba = 0.2
model_rashba = set_checkerboard_model(Delta=Delta,t_0=t_0,tprime=tprime,nspin=2,rashba=rashba)
fig, ax = plt.subplots(2,2, figsize=(10.,8.))
ax = ax.flatten()
k_vec, k_dist, k_node, evals = calcu_band(model_rashba,path)
ax[0].set_title("(b) band structure")
ax[0].set_xlim([0,k_node[-1]])
ax[0].set_xticks(k_node)
ax[0].set_xticklabels(k_lab)
for n in range(len(k_node)):
    ax[0].axvline(x=k_node[n], linewidth=0.5, color='k')
for ib in range(evals.shape[0]):
    ax[0].plot(k_dist,evals[ib],label="band {}".format(ib))
ax[0].legend()

ax[1].set_title("(b) Centers of hWF")
ax[1].set_xlabel(r"k vector along direction 0")
ax[1].set_ylabel(r"Wannier center along direction 1")
nk0,nk1  = 101,11
k_vec   = np.linspace(0,2*np.pi,nk0)
k_node  = [0.0, np.pi, 2*np.pi]
k_label = [r"$0$",r"$\pi$", r"$2\pi$"]
my_array =wf_array(model_rashba,[nk0,nk1])
my_array.solve_on_grid([0.0,0.0])
phi_1 = my_array.berry_phase(occ=[0,1], dir=1, berry_evals=True,contin=True)
ax[1].set_xlim(0,2*np.pi)
ax[1].set_xticks(k_node)
ax[1].set_xticklabels(k_label)
ax[1].set_ylim(-0.5,3+0.5)
for j in range(-1,nk1+1):
    if j==0:
        ax[1].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50,label="TR invariant")
        ax[1].plot(k_vec,float(j)+phi_1[:,0]/(2.0*np.pi),'r-',zorder=-50,label="spin up")
        ax[1].plot(k_vec,float(j)+phi_1[:,1]/(2.0*np.pi),'b-',zorder=-50,label="spin dn")
    else:
        ax[1].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50)
        ax[1].plot(k_vec,float(j)+phi_1[:,0]/(2.0*np.pi),'r-',zorder=-50)
        ax[1].plot(k_vec,float(j)+phi_1[:,1]/(2.0*np.pi),'b-',zorder=-50)
ax[1].legend()


#-------------------- c ----------------------
Delta  = 6.0
t_0    = 1.0
tprime = 0.4
rashba = 0.4
model_rashba = set_checkerboard_model(Delta=Delta,t_0=t_0,tprime=tprime,nspin=2,rashba=rashba)
k_vec, k_dist, k_node, evals = calcu_band(model_rashba,path)
ax[2].set_title("(c) band structure")
ax[2].set_xlim([0,k_node[-1]])
ax[2].set_xticks(k_node)
ax[2].set_xticklabels(k_lab)
for n in range(len(k_node)):
    ax[2].axvline(x=k_node[n], linewidth=0.5, color='k')
for ib in range(evals.shape[0]):
    ax[2].plot(k_dist,evals[ib],label="band {}".format(ib))
ax[2].legend()

ax[3].set_title("(c) Centers of hWF")
ax[3].set_xlabel(r"k vector along direction 0")
ax[3].set_ylabel(r"Wannier center along direction 1")
nk0,nk1  = 101,11
k_vec   = np.linspace(0,2*np.pi,nk0)
k_node  = [0.0, np.pi, 2*np.pi]
k_label = [r"$0$",r"$\pi$", r"$2\pi$"]
my_array =wf_array(model_rashba,[nk0,nk1])
my_array.solve_on_grid([0.0,0.0])
phi_1 = my_array.berry_phase(occ=[0,1], dir=1, berry_evals=True,contin=True)
ax[3].set_xlim(0,2*np.pi)
ax[3].set_xticks(k_node)
ax[3].set_xticklabels(k_label)
ax[3].set_ylim(-0.5,3+0.5)
for j in range(-1,nk1+1):
    if j==0:
        ax[3].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50,label="TR invariant")
        ax[3].plot(k_vec,float(j)+phi_1[:,0]/(2.0*np.pi),'r-',zorder=-50,label="spin up")
        ax[3].plot(k_vec,float(j)+phi_1[:,1]/(2.0*np.pi),'b-',zorder=-50,label="spin dn")
    else:
        ax[3].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50)
        ax[3].plot(k_vec,float(j)+phi_1[:,0]/(2.0*np.pi),'r-',zorder=-50)
        ax[3].plot(k_vec,float(j)+phi_1[:,1]/(2.0*np.pi),'b-',zorder=-50)
ax[3].legend()


fig.tight_layout()
fig.savefig("figure/ex5-10bc.pdf")