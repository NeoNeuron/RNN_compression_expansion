# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
N = 100
dt = 0.01
Wrec=np.random.randn(N,N)/np.sqrt(N)

# g_radius = 20

fig, ax = plt.subplots(1, 1, figsize=(6.5,4), 
                       gridspec_kw=dict(left=0.1, right=0.75, top=0.95, bottom=0.15))
for g_radius in (20,250):
    Weff = (1 - dt)*np.eye(N) + dt*Wrec*g_radius
    eig = np.linalg.eigvals(Weff)
    sns.scatterplot(np.real(eig), np.imag(eig), ax=ax, label=r'$g_{radius}$=%d'%g_radius)
    # ax.plot(np.real(eig), np.imag(eig),'.')
ax.legend(fontsize=14, loc=(1.01,0.65))
ax.axis('scaled')
ax.axvline(1, ls='--', color='red')
ax.set_xlabel(r'Re($\lambda_i$)', fontsize=20)
ax.set_ylabel(r'Im($\lambda_i$)', fontsize=20)
ax.set_xlim(-2)
plt.savefig('image/spectrum_radius.pdf')
# %%
