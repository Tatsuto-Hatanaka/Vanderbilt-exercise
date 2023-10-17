import numpy as np
import matplotlib.pyplot as plt
from pythtb import *

def setmodel(de, t):
    lat = [[1.0]] # 1D
    orb = [[0.0],[0.0]] # s, p
    model = tb_model(1,1,lat,orb)
    model.set_onsite([de/2,-de/2])
    model.set_hop(-1.40*t,0,0,[1])
    model.set_hop( 3.24*t,1,1,[1])
    model.set_hop( 1.84*t,0,1,[1])
    model.set_hop(-1.84*t,1,0,[1])
    return model

def calcu_BP(de,t,nkp=41):
    model = setmodel(de,t)
    evec_array = wf_array(model,[nkp])
    evec_array.solve_on_grid([0.])
    berry_phase = evec_array.berry_phase([0])
    return berry_phase

def calcu_CoWF(de,t):
    model = setmodel(de,t)
    evec_array = wf_array(model,[nkp])

de = 8
tmin, tmax, nt = 0.5, 1.2, 12
#tmin, tmax, nt = 0.8, 0.9, 12
ts = np.linspace(tmin,tmax,nt)
fig,ax = plt.subplots(3,4,figsize=(12,10))
for i,t in enumerate(ts):
    model = setmodel(de,t)
    (k_vec,k_dist,k_node) = model.k_path("full",41)
    evals = model.solve_all(k_vec)
    ax[i//4,i%4].set_title("t = {:.3f}".format(t))
    ax[i//4,i%4].set_xlim(k_dist[0],k_dist[-1])
    #ax[i//4,i-4].set_ylim(min(evals[0])-1,max(evals[1])+1)
    ax[i//4,i%4].set_ylim(-15,10)
    ax[i//4,i%4].axvline(x=k_node[1],linewidth=0.5,color="k")
    ax[i//4,i%4].plot(k_dist,evals[0])
    ax[i//4,i%4].plot(k_dist,evals[1])

fig.savefig("figure/ex3-19-a.pdf")

# t_c ~ 0.9?
tmin, tmax, nt = 0.5, 1.2, 5
ts = np.linspace(tmin,tmax,nt)
print("Berry phase of the lower band")
for i,t in enumerate(ts):
    bp = calcu_BP(de,t)
    #print(bp)
    print("t = {:.3f} : {:.3f} pi".format(t, bp/np.pi))

def solve_Hk0(t,de=8):
    Vss = -1.40*t
    Vpp =  3.24*t
    Vsp =  1.84*t
    Vps = -1.84*t
    Hk0 = np.array([[ de/2+2*Vss,     Vsp+Vps],
                    [    Vsp+Vps, -de/2+2*Vpp]])
    #print(Hk0)
    eval, evec = np.linalg.eigh(Hk0)
    return eval

def solve_Hk0s(ts,de=8):
    evals = np.zeros((len(ts),2))
    for i, t in enumerate(ts):
        evals[i,:] = solve_Hk0(t,de)
    return evals

ts = np.linspace(0.5,1.2,31)
evals = solve_Hk0s(ts)

fig, ax = plt.subplots()
# plt.figure(figsize=(5,4))
ax.set_title("Eigenvalues of 2 bands for k = 0")
ax.set_xlim(min(ts),max(ts))
d = 0.5
ax.set_ylim(min(evals[:,0])-d,max(evals[:,1])+d)
ax.set_xlabel("t")
ax.set_ylabel("Eigenvalue")
ax.plot(ts,evals[:,0],label="lower")
ax.plot(ts,evals[:,1],label="upper")
ax.legend()
# plt.show()

fig.savefig("figure/ex3-19-b.pdf")