from pythtb import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def set_bulk_model(lam):
    t     = -1.0 # average hopping
    Delta = -0.4*np.cos(lam) # site energy alternation
    del_t = -0.3*np.sin(lam) # bond strength alternation
    lat   = [[1.0]]
    orb   = [[0.0],[0.5]]
    bulk_model = tb_model(1,1,lat,orb)
    bulk_model.set_onsite([Delta,-Delta])
    bulk_model.set_hop(t+del_t, 0, 1, [0])
    bulk_model.set_hop(t-del_t, 1, 0, [1])
    return bulk_model

def set_fin_model(n_cell,en_shift,lam):
    # set parameters of model
    t     = -1.0 # average hopping
    Delta = -0.4*np.cos(lam) # site energy alternation
    del_t = -0.3*np.sin(lam) # bond strength alternation
    lat   = [[1.0]]
    orb   = [[0.0],[0.5]]
    bulk_model = tb_model(1,1,lat,orb)
    bulk_model.set_onsite([Delta,-Delta])
    bulk_model.set_hop(t+del_t, 0, 1, [0])
    bulk_model.set_hop(t-del_t, 1, 0, [1])
    # cut chain of length n_cell and shift energy on last site
    finite_model = bulk_model.cut_piece(n_cell,0)
    finite_model.set_onsite(en_shift,ind_i=2*n_cell-1,mode='add')
    return finite_model


ef      = 0.18
n_cell  = 20
n_orb   = 2*n_cell
n_param = 101
params   = np.linspace(0.,1.,n_param)
eig_sav  = np.zeros((n_orb,n_param),dtype=float)
xbar_sav = np.zeros((n_orb,n_param),dtype=float)
nocc_sav = np.zeros((n_param),dtype=int)
surf_sav = np.zeros((n_param),dtype=float)
count    = np.zeros((n_orb),dtype=float)
# for ex4-12
xbar_sav2 = np.zeros((n_orb,n_param),dtype=float)
surf_sav2 = np.zeros((n_param),dtype=float)
count2    = np.zeros((n_orb),dtype=float)

mpl.rc('font',size=10) # set global font size
fig,ax=plt.subplots(3,2,figsize=(7.,6.),gridspec_kw={'height_ratios':[2,1,1]},sharex="col")

# loop over two cases: vary surface site energy, or vary lambda
for mycase in ['surface energy','lambda']:
    if mycase == 'surface energy':
        (ax0,ax1,ax2) = ax[:,0] # axes for plots in left panels
        ax0.text(-0.30,0.90,'(a)',size=12.,transform=ax0.transAxes)
        lmbd     = 0.15*np.pi*np.ones((n_param),dtype=float)
        en_shift = -3.0+6.0*params
        abscissa = en_shift
    elif mycase == 'lambda':
        (ax0,ax1,ax2) = ax[:,1] # axes for plots in right panels
        ax0.text(-0.30,0.90,'(b)',size=12.,transform=ax0.transAxes)
        lmbd     = params*2.*np.pi
        en_shift = 0.2*np.ones((n_param),dtype=float)
        abscissa = params

    # loop over parameter values
    for j in range(n_param):
        my_model = set_fin_model(n_cell, en_shift[j], lmbd[j])
        eval, evec = my_model.solve_all(eig_vectors=True)
        # print(evec.shape)
        nocc = (eval < ef).sum()
        ovec = evec[:nocc,:]
        xbar_sav[:nocc,j] = my_model.position_hwf(ovec,0)
        # get electron count on each site
        # convert to charge (2 for spin; unit nuclear charge per site)
        # compute surface charge down to depth of 1/3 of chain
        for i in range(n_orb):
            count[i] = np.real(np.vdot(evec[:nocc,i], evec[:nocc,i]))
        charge  = -2.*count+1.
        n_cut   = int(0.67*n_orb)
        surf_sav[j]  = 0.5*charge[n_cut-1]+charge[n_cut:].sum()
        nocc_sav[j]  = nocc
        eig_sav[:,j] = eval

        # ex 4-12
        nkp = 41
        bulk_model = set_bulk_model(lmbd[j])
        wf_bulk = wf_array(bulk_model,[nkp])
        wf_bulk.solve_on_grid([0.0])
        eval, evec = bulk_model.solve_all(np.linspace(-1/2,1/2,nkp), eig_vectors=True)
        occ0 = (abs(evec[0, eval[0]<ef, :])**2).sum()/nkp
        occ1 = (abs(evec[1, eval[1]<ef, :])**2).sum()/nkp
        pos0 = wf_bulk.berry_phase(occ=[0],contin=True,berry_evals=True)[0]/(2*np.pi)
        pos1 = wf_bulk.berry_phase(occ=[1],contin=True,berry_evals=True)[0]/(2*np.pi)
        # if abs(j-50)<5: print(j,occ0,occ1,pos0,pos1)
        p = 1*0.5 - 2*(occ0*pos0 + occ1*pos1)
        surf_x = int(n_cell*0.67)
        n_wan = sum(xbar_sav[:nocc,j] > surf_x)
        delQ = -2*n_wan + 2*(n_cell-surf_x)
        if mycase=="lambda": print(j, delQ)
        surf_sav2[j] = p + delQ


    # plot the figures
    ax0.set_xlim(0.,1.)
    ax0.set_ylim(-2.8,2.8)
    ax0.set_ylabel(r"Band energy")
    ax0.axhline(y=ef,color='k',linewidth=0.5)
    for n in range(n_orb):
        ax0.plot(abscissa, eig_sav[n,:], color='k')

    ax1.set_xlim(0.,1.)
    ax1.set_ylim(n_cell-4.6,n_cell+0.4)
    ax1.set_yticks(np.linspace(n_cell-4,n_cell,5))
    #ax1.set_ylabel(r"$\bar{x}$")
    ax1.set_ylabel(r"Wannier centers")
    for j in range(n_param):
        nocc=nocc_sav[j]
        ax1.scatter([abscissa[j]]*nocc,xbar_sav[:nocc,j],color='k',
            s=3.,marker='o',edgecolors='none')

    ax2.set_ylim(-5.2,5.2)
    ax2.set_yticks([-2.,-1.,0.,1.,2.])
    ax2.set_ylabel(r"Surface charge")
    if mycase == 'surface energy':
        ax2.set_xlabel(r"Surface site energy")
    elif mycase == 'lambda':
        ax2.set_xlabel(r"$\lambda/2\pi$")
    ax2.set_xlim(abscissa[0],abscissa[-1])
    ax2.scatter(abscissa,surf_sav,color='k',s=3.,marker='o',edgecolors='none')
    ax2.scatter(abscissa,surf_sav2,color='r',s=3.,marker='o',edgecolors='none')

    # vertical lines denote surface state at right end crossing the
    # Fermi energy
    for j in range(1,n_param):
        if nocc_sav[j] != nocc_sav[j-1]:
            n      = min(nocc_sav[j],nocc_sav[j-1])
            frac   = (ef-eig_sav[n,j-1])/(eig_sav[n,j]-eig_sav[n,j-1])
            a_jump = (1-frac)*abscissa[j-1]+frac*abscissa[j]
            if mycase == 'surface energy' or nocc_sav[j] < nocc_sav[j-1]:
                ax0.axvline(x=a_jump, color='k', linewidth=0.5)
                ax1.axvline(x=a_jump, color='k', linewidth=0.5)
                ax2.axvline(x=a_jump, color='k', linewidth=0.5)

fig.tight_layout()
plt.subplots_adjust(left=0.12,wspace=0.4)
fig.savefig("figure/ex4-12.pdf")