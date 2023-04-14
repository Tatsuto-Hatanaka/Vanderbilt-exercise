from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

def set_model(t1,t2):
    lat = [[1,0],[1/2,np.sqrt(3)/2]]
    orb = [[0,0],[1/2,0],[0,1/2]]
    model = tb_model(2,2,lat,orb)
    model.set_onsite([0,0,0])
    model.set_hop(t1+1j*t2,0,1,[0,0])
    model.set_hop(t1+1j*t2,1,2,[0,0])
    model.set_hop(t1+1j*t2,2,0,[0,0])
    model.set_hop(t1+1j*t2,0,1,[-1,0])
    model.set_hop(t1+1j*t2,1,2,[1,-1])
    model.set_hop(t1+1j*t2,2,0,[0,1])
    return model

def calcu_band(my_model,path):
    # generate k-point path and labels and solve Hamiltonian
    (k_vec,k_dist,k_node) = my_model.k_path(path,121,report=False)
    evals = my_model.solve_all(k_vec)
    return k_vec, k_dist, k_node, evals

path = [[0.,0.],[2./3.,1./3.],[.5,.5],[1./3.,2./3.], [0.,0.]]
k_lab = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')

colors = ['#FF0000', '#FF3F00', '#FF7F00', '#FFBF00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#2E2B5F', '#8B00FF', '#FF00FF']
cmap = LinearSegmentedColormap.from_list('mycmap', colors)

fig, ax = plt.subplots(1,1, figsize=(4.,3.))

#------------------- a ---------------------
t1 = 1.0
t2s = np.linspace(0,1,11)
ax.set_title("$t_1 = {}$".format(t1))
for it2, t2 in enumerate(t2s):
    my_model = set_model(t1=t1,t2=t2)
    k_vec, k_dist, k_node, evals = calcu_band(my_model,path)
    if it2==0:
        ax.set_xlim([0,k_node[-1]])
        ax.set_xticks(k_node)
        ax.set_xticklabels(k_lab)
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k')
    # if it2%2!=0: continue
    for ib in range(evals.shape[0]):
        if ib==0:
            ax.plot(k_dist,evals[ib],color=cmap(it2/(len(colors)-1)),label="$t_2={}$".format(t2.round(2)))
        else:
            ax.plot(k_dist,evals[ib],color=cmap(it2/(len(colors)-1)))
ax.legend()
fig.savefig("figure/ex5-4a.pdf")



#------------------- b ---------------------
t1       = 1.0
t2s      = [0.0, 0.5]
nk0,nk1  = 101,11
num_bands = 3
k_vec = np.linspace(0,2*np.pi,nk0)
k_node = [0.0, np.pi, 2*np.pi]
k_label=[r"$0$",r"$\pi$", r"$2\pi$"]

fig, ax = plt.subplots(1,3, figsize=(12.,4.))
for it2, t2 in enumerate(t2s):
    my_model = set_model(t1=t1,t2=t2)
    my_array=wf_array(my_model,[nk0,nk1])
    my_array.solve_on_grid([0.0,0.0])
    for ib in range(num_bands):
        phi_1=my_array.berry_phase(occ=[ib], dir=1, contin=True)
        # print(phi_1.shape)
        ax[ib].set_title("Center of hWF of band {}".format(ib))
        ax[ib].set_xlabel(r"k vector along direction 0")
        if ib==0: ax[ib].set_ylabel(r"Wannier center along direction 1")
        ax[ib].set_xlim(0,2*np.pi)
        ax[ib].set_xticks(k_node)
        ax[ib].set_xticklabels(k_label)
        ax[ib].set_ylim(-0.5,3+0.5)
        for j in range(-1,nk1+1):
            if j==0 and it2==0:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'k-',zorder=-50,label="$t_2={}$".format(t2))
            elif j==0 and it2==1:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'r-',zorder=-50,label="$t_2={}$".format(t2))
            elif j!=0 and it2==0:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'k-',zorder=-50)
            elif j!=0 and it2==1:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'r-',zorder=-50)
        ax[ib].legend()
fig.savefig("figure/ex5-4b.pdf")



#------------------- c ---------------------
fig, ax = plt.subplots(1,2, figsize=(8.,4.))
comb_bands = [[0,1],[2]]
bands_label = ["0+1", "2"]

for it2, t2 in enumerate(t2s):
    my_model = set_model(t1=t1,t2=t2)
    my_array=wf_array(my_model,[nk0,nk1])
    my_array.solve_on_grid([0.0,0.0])
    for ib in range(len(comb_bands)):
        phi_1=my_array.berry_phase(occ=comb_bands[ib], dir=1, contin=True)
        # print(phi_1.shape)
        ax[ib].set_title("Center of hWF of band "+bands_label[ib])
        ax[ib].set_xlabel(r"k vector along direction 0")
        if ib==0: ax[ib].set_ylabel(r"Wannier center along direction 1")
        ax[ib].set_xlim(0,2*np.pi)
        ax[ib].set_xticks(k_node)
        ax[ib].set_xticklabels(k_label)
        ax[ib].set_ylim(-0.5,3+0.5)
        for j in range(-1,nk1+1):
            if j==0 and it2==0:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'k-',zorder=-50,label="$t_2={}$".format(t2))
            elif j==0 and it2==1:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'r-',zorder=-50,label="$t_2={}$".format(t2))
            elif j!=0 and it2==0:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'k-',zorder=-50)
            elif j!=0 and it2==1:
                ax[ib].plot(k_vec,float(j)+phi_1/(2.0*np.pi),'r-',zorder=-50)
        ax[ib].legend()
fig.savefig("figure/ex5-4c.pdf")