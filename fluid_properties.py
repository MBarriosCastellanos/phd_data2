# %% =======================================================================
# Import libraries 
# ========================================================================== 
import autograd.numpy as np                     # mathematics
from autograd import jacobian, hessian
from scipy.optimize import minimize     # optimization arguments
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
d_s_a = {                               # arguments transform in h5 file
  'compression':'gzip', 'shuffle':True, 'fletcher32':True }
sep = os.path.sep                       # path separator independin of OS
print('Basic Libraries imported')
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8

# %% =======================================================================
# Import data
# ==========================================================================
path = os.path.join('data', 'viscosity.csv') 
df = pd.read_csv(path)# mu Pa-s Tin C
df.plot(x='t', y='mu', style='.')
path2 = os.path.join('data', 'viscosityp100.csv') 
df2 = pd.read_csv(path2)# mu Pa-s Tin C
df2.plot(x='t', y='mu', style='.')

# %% =======================================================================
# polynomial prediction
# ========================================================================== 
# Polynomial model function
MU_poly = lambda t, coef: np.sum(np.array([
  coef[i]*t**(len(coef)- i- 1) for i in range(len(coef))]), axis=0)
coef = np.polyfit(df.t, df.mu, 5)
mu_pred = MU_poly(df.t, coef)
print('coef = ', coef)
print('mae = ', 100*mean_absolute_error(df.mu, mu_pred))
print('r2 = ',r2_score(df.mu, mu_pred))

# %% =======================================================================
# exponential adjust
# ==========================================================================
MU_exp = lambda t, coef: coef[0]*np.exp(-coef[1]*t**coef[2])
def obj(coef):
  mu_pred = MU_exp(np.array(df.t), coef) 
  return np.sum((  np.array(df.mu)- mu_pred)**2)
jac = jacobian(obj);  hes = hessian(obj)
res = minimize(obj, [0.1, 0.1, 0.1], jac=jac, hess=hes, method='trust-exact')
coef_exp = res.x
mu_pred = MU_exp(df.t, coef_exp)
print('residual  =', res.fun, 'sucess = ', res.success)
print('coef = ', coef_exp)
print('mae = ', 100*mean_absolute_error(df.mu, mu_pred))
print('r2 = ',r2_score(df.mu, mu_pred))

# %% =======================================================================
# plot prediction
# ========================================================================== 
MU1 = lambda T: 0.00026436*T**4 - 0.04864*T**3 + 3.436*T**2 - \
  114.28*T + 1610.3
t = np.linspace(df.t.min(), df.t.max(), num=1000)
t2 = np.linspace(df2.t.min(), df2.t.max(), num=1000)

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(df2.t, df2.mu, '.', label='óleo P100', color='gray',markersize=5)
ax.plot(df.t, 1000*df.mu, '.', label='óleo P47', color='k' ,markersize=5)

#ax.plot(t, 1000*MU_poly(t, coef), '-o', label='poly', color='gray', alpha=0.9,
#  linewidth=2,  markersize=7,  markevery=100)
ax.plot(t2, MU1(t2), '-o', label='ajuste P100' , color='r',
  linewidth=2, alpha=0.7, markersize=5, markevery=100,markerfacecolor='none')
ax.plot(t, 1000*MU_exp(t, coef_exp), '-s', 
  label='ajuste P47' , color='r',
  linewidth=2, alpha=0.7, markersize=5, markevery=100,markerfacecolor='none')
#plt.yscale('log')
ax.set_xlabel('temperatura [$\mathrm{^{o} C}$]')
ax.set_ylabel('viscosidade [cP]')
ax.legend()
fig.savefig('result/viscosity.pgf')


# %% =======================================================================
# plot prediction
# ========================================================================== 
MU1 = lambda T: 0.00026436*T**4 - 0.04864*T**3 + 3.436*T**2 - \
  114.28*T + 1610.3
t = np.linspace(df.t.min(), df.t.max(), num=1000)
t2 = np.linspace(df2.t.min(), df2.t.max(), num=1000)

fig, ax = plt.subplots(figsize=(3.1,1.7))
ax.plot(df.t, 1000*df.mu, '.', label='data', color='k' ,markersize=4)

ax.plot(t, 1000*MU_exp(t, coef_exp), '-s', 
  label='curve adjusted' , color='r',
  linewidth=2, alpha=0.7, markersize=7, markevery=100,markerfacecolor='none')
#plt.yscale('log')
ax.set_xlabel('temperature [$\mathrm{^{o} C}$]')
ax.set_ylabel('viscosity [cP]')
ax.legend()
fig.savefig('result/viscosity2.pgf')
# %% =======================================================================
# Import data density
# ==========================================================================
path = os.path.join('data', 'density.csv') 
df = pd.read_csv(path)# mu Pa-s Tin C
df.rho = df.rho*1000
df.plot(x='t', y='rho', style='.')

# %% =======================================================================
# polynomial prediction
# ========================================================================== 
# Polynomial model function
RHO = lambda t, coef: np.sum(np.array([
  coef[i]*t**(len(coef)- i- 1) for i in range(len(coef))]), axis=0)
coef = np.polyfit(df.t, df.rho, 1)
rho_pred = MU_poly(df.t, coef)
print('coef = ', coef)
print('mae = ', 100*mean_absolute_error(df.rho, rho_pred))
print('r2 = ',r2_score(df.rho, rho_pred))

# %% =======================================================================
# plot prediction
# ========================================================================== 
t = np.linspace(df.t.min(), df.t.max(), num=1000)
plt.plot(df.t, df.rho, '.', label='data', color='k')
plt.plot(t, RHO(t, coef), '--', label='exp' , color='r',
  linewidth=2, alpha=0.5,
  markerfacecolor='none')
plt.xlabel('temperature [C]')
plt.ylabel('viscosity [cP]')
plt.legend()

# %% =======================================================================
# save model
# ==========================================================================
path = os.path.join('result', 'model2.h5') 
model  = {} if len(glob.glob(path))==0 else h5todict(path)
model['mu'] = coef_exp
model['rho'] = coef
dicttoh5(model, path, create_dataset_args=d_s_a)
