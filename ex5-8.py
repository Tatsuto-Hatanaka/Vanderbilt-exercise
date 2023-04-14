from pythtb import *
import matplotlib.pyplot as plt
import numpy as np

def set_2l_haldane_model(delta,t,t2,tl):
    lat = [[1,0,0],[1/2,np.sqrt(3)/2,0],[0,0,1]]
    orb = [[1/3,1/3,0],[1/2,1/2,0],[1/3,1/3,0.5],[1/2,1/2,0.5]]
    model = tb_model(2,3,lat,orb)
    model.set_onsite([-delta,delta]*2)
    for r in [[0,0,0], [-1,0,0], [0,-1,0]]:
        model.set_hop(t,0,1,r)
        model.set_hop(t,2,3,r)
    for r in [[1,0,0], [-1,1,0], [0,-1,0]]:
        model.set_hop(t2*1.j,0,0,r)
        model.set_hop(t2*1.j,2,2,r)
    for r in [[-1,0,0], [1,-1,0], [0,1,0]]:
        model.set_hop(t2*1.j,1,1,r)
        model.set_hop(t2*1.j,3,3,r)
    for io in range(2):
        model.set_hop(tl,io,io+2,[0,0,0])
    return model

def set_2l_kanemele_model(esite,thop,soc,rashba,tl):
    lat=[[1.0,0.0,0],[0.5,np.sqrt(3.0)/2.0,0],[0,0,1]]
    orb=[[1./3.,1./3.,0],[2./3.,2./3.,0],[1./3.,1./3.,0.5],[2./3.,2./3.,0.5]]
    model = tb_model(2,3,lat,orb,nspin=2)
    sigma_x=np.array([0.,1.,0.,0])
    sigma_y=np.array([0.,0.,1.,0])
    sigma_z=np.array([0.,0.,0.,1])
    model.set_onsite([esite,(-1.0)*esite]*2)
    for r in [[0,0,0],[0,-1,0],[-1,0,0]]:
        model.set_hop(thop, 0, 1, r)
        model.set_hop(thop, 2, 3, r)
    for ir, r in enumerate([[0,1,0],[1,0,0],[1,-1,0]]):
        for io in range(len(orb)):
            model.set_hop((-1)**(ir+io+1)*(-1.j)*soc*sigma_z,io,io,r)
    r3h =np.sqrt(3.0)/2.0
    for io in [0,2]:
        model.set_hop(1.j*rashba*( 0.5*sigma_x-r3h*sigma_y), io, io+1, [0, 0,0], mode="add")
        model.set_hop(1.j*rashba*(-1.0*sigma_x            ), io, io+1, [0,-1,0], mode="add")
        model.set_hop(1.j*rashba*( 0.5*sigma_x+r3h*sigma_y), io, io+1, [-1,0,0], mode="add")
    for io in range(2):
        model.set_hop(tl,io,io+2,[0,0,0])
    return model

def calcu_band(my_model,path):
    # generate k-point path and labels and solve Hamiltonian
    (k_vec,k_dist,k_node) = my_model.k_path(path,121,report=False)
    evals = my_model.solve_all(k_vec)
    return k_vec, k_dist, k_node, evals

fig, ax = plt.subplots(2,2,figsize=(10.,8.))



#------------------ a -------------------

delta = 0.7
t     = -1.0
t2    = -0.3
tl    = 0.2 # hopping between layer
model = set_2l_haldane_model(delta,t,t2,tl)

# band
path = [[0.,0.],[2./3.,1./3.],[.5,.5],[1./3.,2./3.], [0.,0.]]
k_lab = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')
k_vec, k_dist, k_node, evals = calcu_band(model,path)
ax[0,0].set_title("(a) Band str. of 2L Haldane model")
ax[0,0].set_ylabel("Energy (eV)")
ax[0,0].set_xlim([0,k_node[-1]])
ax[0,0].set_xticks(k_node)
ax[0,0].set_xticklabels(k_lab)
for n in range(len(k_node)):
    ax[0,0].axvline(x=k_node[n], linewidth=0.5, color='k')
for ib in range(evals.shape[0]):
    ax[0,0].plot(k_dist,evals[ib],label="band {}".format(ib))
ax[0,0].legend()

# hWF centers
ax[0,1].set_title("(a) Centers of hWF in 2L Haldane model")
ax[0,1].set_xlabel(r"k vector along direction 0")
ax[0,1].set_ylabel(r"Wannier center along direction 1")
nk0,nk1  = 101,11
k_vec = np.linspace(0,2*np.pi,nk0)
k_node = [0.0, np.pi, 2*np.pi]
k_label=[r"$0$",r"$\pi$", r"$2\pi$"]
my_array=wf_array(model,[nk0,nk1])
my_array.solve_on_grid([0.0,0.0])
phi_1=my_array.berry_phase(occ=[0,1], dir=1, contin=True, berry_evals=True)
ax[0,1].set_xlim(0,2*np.pi)
ax[0,1].set_xticks(k_node)
ax[0,1].set_xticklabels(k_label)
ax[0,1].set_ylim(-0.5,3+0.5)
for j in range(-1,nk1+1):
    if j==0:
        ax[0,1].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50,label="band 0+1")
        # for i in [0,1]:
        #     ax[0,1].plot(k_vec,float(j)+phi_1[:,i]/(2.0*np.pi),zorder=-50,label="band {}".format(i))
    else:
        ax[0,1].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50)
        # for i in [0,1]:
        #     ax[0,1].plot(k_vec,float(j)+phi_1[:,i]/(2.0*np.pi),zorder=-50)
ax[0,1].legend()




#------------------ b -------------------

ax[1,0].set_title("(b) Band str. of 2L Kane-Mele model")
# esite  = 2.5 # even
esite  = 1.0 # odd
thop   = 1.0
soc    = 0.6*thop*0.5
rashba = 0.25*thop
tl     = 0.2
model = set_2l_kanemele_model(esite,thop,soc,rashba,tl)

path = [[0.,0.],[2./3.,1./3.],[.5,.5],[1./3.,2./3.], [0.,0.]]
k_lab = (r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')
k_vec, k_dist, k_node, evals = calcu_band(model,path)
ax[1,0].set_ylabel("Energy (eV)")
ax[1,0].set_xlim([0,k_node[-1]])
ax[1,0].set_xticks(k_node)
ax[1,0].set_xticklabels(k_lab)
for n in range(len(k_node)):
    ax[1,0].axvline(x=k_node[n], linewidth=0.5, color='k')
for ib in range(evals.shape[0]):
    ax[1,0].plot(k_dist,evals[ib],label="band {}".format(ib))
ax[1,0].legend()


# hWF centers
ax[1,1].set_title("(b) Centers of hWF in 2L Kane-Mele model")
ax[1,1].set_xlabel(r"k vector along direction 0")
ax[1,1].set_ylabel(r"Wannier center along direction 1")
nk0,nk1  = 101,11
k_vec = np.linspace(0,2*np.pi,nk0)
k_node = [0.0, np.pi, 2*np.pi]
k_label=[r"$0$",r"$\pi$", r"$2\pi$"]
my_array=wf_array(model,[nk0,nk1])
my_array.solve_on_grid([0.0,0.0])
phi_1=my_array.berry_phase(occ=[0,1,2,3], dir=1, contin=True, berry_evals=True)
ax[1,1].set_xlim(0,2*np.pi)
ax[1,1].set_xticks(k_node)
ax[1,1].set_xticklabels(k_label)
ax[1,1].set_ylim(-0.5,3+0.5)
for j in range(-1,nk1+1):
    if j==0:
        ax[1,1].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50,label="band 0~3")
        # for i in [0,1,2,3]:
        #     ax[1,1].plot(k_vec,float(j)+phi_1[:,i]/(2.0*np.pi),zorder=-50,label="band {}".format(i))
    else:
        ax[1,1].plot(k_vec,float(j)+phi_1.sum(axis=1)/(2.0*np.pi),'k-',zorder=-50)
        # for i in [0,1,2,3]:
        #     ax[1,1].plot(k_vec,float(j)+phi_1[:,i]/(2.0*np.pi),zorder=-50)
ax[1,1].legend()


fig.tight_layout()
fig.savefig("figure/ex5-8.pdf")