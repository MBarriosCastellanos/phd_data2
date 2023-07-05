# %% =======================================================================
# Import libraries 
# ========================================================================== 
import autograd.numpy as np                     # mathematics
from autograd import jacobian, hessian
from scipy.optimize import minimize     # optimization arguments
from sklearn.metrics import mean_absolute_error, r2_score
from nptdms import TdmsFile
import os
import re
import glob
from functions import Sample, Sample_vib, printProgressBar
from functions import combined_filter, df_SI
import pandas as pd
import matplotlib.pyplot as plt
from functions import time_elapsed
from thermo.chemical import Chemical            # chemical properties
from functions_model import MU, RHO
d_s_a = {                               # arguments transform in h5 file
  'compression':'gzip', 'shuffle':True, 'fletcher32':True }
print('Basic Libraries imported')
import seaborn as sns
from scipy.constants import g as g
sep = os.path.sep                       # path separator independin of OS
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files


# %% =======================================================================
# Classes and Functions
# =========================================================================
RE = lambda rho, d, q, mu: 4*rho*q/(mu*np.pi*d)
class variables(object):
  def __init__(self, df):
    df = df_SI(df.copy())                      # put dataframe in SI
    self.n = 9                          # number of pump stages
    self.D = 0.108                      # [m] pump rotor diameter
    self.curve_kind = np.array(df.curve_kind)
    self.phase = 0*(self.curve_kind<2) + 1*(self.curve_kind>=2) 
    self.T = np.array(df.T_2)
    self.tau = np.array(df.ESP_torque)
    self.i = np.array(df.ESP_current)
    self.w = np.array(df.ESP_rotation)  # rotation 
    self.curve = np.array(df.i_curve)
    self.dp_ch = np.array(df.DP_choke)
    self.dp_he = np.array(df.DP_he)
    self.p0 = np.array(df.P_0)
    self.p1 = np.array(df.P_1)
    self.ch = np.array(df.choke_esp)
    self.w_b = np.array(df.booster_oil)
    # properties -------------------------------------------------------
    self.rho_w = np.zeros(len(df)); self.mu_w = np.zeros(len(df))
    for i in range(len(self.T)):
      self.rho_w[i] = Chemical('Water',T = self.T[i]+ 273.15).rho   
      self.mu_w[i] =  Chemical('Water',T = self.T[i]+ 273.15).mu 
    self.mu_o = MU(self.T)
    ## input water line -------------------------------------------------
    self.q_win = np.array(df['Q_water'])/self.rho_w   # Q water moved by water pump
    # input mixed line -------------------------------------------------
    self.wc_mix  = np.array(df['Watercut']   )
    self.rho_mix = np.array(df['Density_oil'])
    self.q_mix   = np.array(df['Q_oil'])/self.rho_mix
    self.mu_mix  = self.mu_w*self.wc_mix + self.mu_o*(1 - self.wc_mix) 
    self.rho_o = (self.rho_mix - self.rho_w*(self.wc_mix)
      )/(1 - self.wc_mix)
    ## Total Q_water and Q_oil ------------------------------------------
    q = self.q_win + self.q_mix
    self.q_water = self.q_mix*self.wc_mix + self.q_win  # Total Q water
    self.q_oil = self.q_mix*(1- self.wc_mix)       # total Q oil
    wc  = self.q_water/q                 # Water cut
    rho = (self.rho_w*self.q_win + self.rho_mix*self.q_mix )/q
    mu  = (self.mu_w*self.q_water + self.mu_o*self.q_oil )/q
    # flow calculation -------------------------------------------------
    self.q   =  (self.curve_kind > 1)*self.q_mix + \
                (self.curve_kind == 0)*self.q_win + \
                (self.curve_kind == 1)*self.q_mix #+ \
                #(self.curve_kind > 2)*q  
    self.rho =  (self.curve_kind > 1)*self.rho_mix + \
                (self.curve_kind == 0)*self.rho_w   + \
                (self.curve_kind == 1)*self.rho_mix #+ \
                #(self.curve_kind > 2)*rho
    self.mu  =  (self.curve_kind > 1)*self.mu_o + \
                (self.curve_kind == 0)*self.mu_w   + \
                (self.curve_kind == 1)*self.mu_w   #+ \
                #(self.curve_kind > 2)*mu
    self.wc  =  (self.curve_kind >1 )*self.wc_mix       + \
                (self.curve_kind == 0)*np.ones_like(wc)  + \
                (self.curve_kind == 1)*self.wc_mix       #+ \
                #(self.curve_kind > 2)*wc   
    # pressure calculation ---------------------------------------------
    self.dp  = np.array(df.P_9 - df.P_1)/self.n        # [Pa]
    self.h   = self.dp/(g*self.rho)
    # dimensionless ----------------------------------------------------
    self.phi = self.q/(self.w*self.D**3)
    self.Psi = self.dp/(self.rho*self.w**2*self.D**2)
    self.Psi_w = self.dp/(self.rho_w*self.w**2*self.D**2)
    self.Psi_o = self.dp/(self.rho_o*self.w**2*self.D**2)
    self.X   = self.mu/(self.rho*self.w*self.D**2)
    self.Xi  = self.D*self.mu/(self.q*self.rho)
    self.Xi_w  = self.D*self.mu_w/(self.q*self.rho_w)
    self.Xi_o  = self.D*self.mu_o/(self.q*self.rho_o)
    self.Pi = self.tau/(self.n*self.rho*self.w**2*self.D**5)
    self.Pi_w = self.tau/(self.n*self.rho*self.w**2*self.D**5)
    self.Pi_o = self.tau/(self.n*self.rho*self.w**2*self.D**5)
    # power calculation ------------------------------------------------
    self.PE = self.i*380*0.88*(3**0.5)/self.n # electric power
    self.PM = self.tau*self.w/self.n          # mechanical power
    self.PH = self.q*self.dp                  # hydraulic power
    self.eta = self.PH/self.PE                # total efficiency
    self.eta_m = self.PM/self.PE              # mechanical efficiency
    self.eta_h = self.PH/self.PM              # hydraulic efficiency

# %% =======================================================================
# list curves
# ========================================================================== 
curves = sorted(glob.glob('data/ss*'))
curves_kind = []
k = 0
df_inf = pd.DataFrame(columns=['k', 'i', 'curve', 'point', 'curve_kind'])
for i, curve in enumerate(curves):
  print(('%s %s'%(i, curve)).center(70, '.'))
  curve_list = curve.split(sep)[-1].split('_')
  curve_kind = '_'.join(curve_list[:3])
  curves_kind.append(curve_kind)
  for path in sorted(glob.glob(curve + '/e*')):
    df_inf.loc[k, :] = [str(k), str(i), curve, 
      path.split(sep)[-1], curve_kind] 
    k+=1
curves_kind_unique = np.sort(np.unique(curves_kind))
print('k_max = ', k); k_max = k

# %% =======================================================================
# get process curve
# ==========================================================================
k = 0
paths_del = []
for i, curve in enumerate(curves):
  for path in sorted(glob.glob(curve + '/e*')):
    printProgressBar(k + 1, k_max, );   print('\n')
    exp = Sample(path);                 exp.set_data()
    df = pd.DataFrame(exp.data)
    if np.shape(df)[0]!=3750 or np.shape(df)[1]!=33:
      paths_del.append(path)
      print(' incomplete path '.center(70, '█'))  
    #exp_vib = Sample_vib(path);     exp_vib.set_data()
    #df_vib = pd.DataFrame(exp_vib.data)
    #if np.shape(df_vib)[0]!=768000 or np.shape(df_vib)[1]!=8:
    #  paths_del.append(path)
    #  print(' incomplete path '.center(70, '█'))  
    df['ESP_rotation'] = combined_filter(df['time'], df['ESP_rotation'], 
      10, 10/3, 20)
    df = df.iloc[500:-500]
    df.index = range(len(df.index))
    curve_kind = curves_kind[i]
    l = len(df)//25
    df_i = pd.DataFrame(columns=df.columns)
    for j in range(l):
      df_j = pd.concat([df.loc[j*25   :j*25+12, :].mean().T,
                        df.loc[j*25+12:j*25+24, :].mean().T], axis=1).T
      df_i = pd.concat([df_i, df_j], axis=0, ignore_index=True)
    df_i['i_curve'] = i
    df_i['k'] = k
    df_i['curve_kind'] = np.where(curves_kind_unique==curve_kind)[0][0]
    df_i['time'] = np.round(df_i['time'] - df_i['time'][0], 2)
    DF = df_i if k==0 else pd.concat([DF, df_i], axis=0, ignore_index=True)
    df_mean_i = pd.DataFrame(df_i.mean()).T
    df_mean = df_mean_i if k==0 else pd.concat( 
      [df_mean, df_mean_i], axis=0, ignore_index=True)
    k+=1
print('last k is =', k)

# %% =======================================================================
# get dataframes
# ==========================================================================
df_w  = df_mean[df_mean.curve_kind==0].copy()
df_w1  = df_mean[df_mean.curve_kind==1].copy()
df_o  = df_mean[df_mean.curve_kind==2].copy()
df_o1 = df_mean[df_mean.curve_kind==3].copy()
#DF_o2 = DF[DF.curve_kind==3].copy(); df_o2 = df_mean[df_mean.curve_kind==4].copy()


# %% =======================================================================
# analyses curves 
# ==========================================================================
case = range(36)
var = variables(df_mean)
phi_test = np.linspace(0, var.phi.max(), 10000)
q_test = np.linspace(0, var.q.max(), 10000)
COEF = []
BEP_data = [] # mu, omega, q_bep, phi_bep
for j, i in enumerate(case):
  j = 0
  # variables selected ----------------------------------------------------
  var = variables(df_mean[df_mean.i_curve==i])
  fig, ax =  plt.subplots(2, 2, figsize=(12, 9))
  # first plot-------------------------------------------------------------
  ax[0, 0].set_xlim([0, 3600*var.q.max()*1.01])
  ax[0, 0].plot(var.q*3600, var.dp*1e-5*var.n, '.', color='k')
  ax[0, 0].set_xlabel('$q$ [$\\mathrm{m^3/h}$]')
  ax[0, 0].set_ylabel('$\\Delta p$ [bar]')
  # second plot -----------------------------------------------------------
  coeff = np.polyfit(var.phi, var.eta_h,3)
  curve_eff = np.poly1d(coeff)
  phi_bep = phi_test[np.argmax(curve_eff(phi_test))]
  # _______________________________________________________________________
  ax[0, 1].set_xlim([0, var.phi.max()*1.01])
  ax[0, 1].plot(var.phi, var.Psi, '.', color='k')
  ax1 = ax[0, 1].twinx()
  ax1.plot(var.phi, var.eta_h, 'x', color='g')
  ax1.plot(phi_test, curve_eff(phi_test), '--', color='g')
  ax[0, 1].set_xlabel('$\\phi$')
  ax[0, 1].set_ylabel('$\\psi$')
  ax1.set_ylabel('$\\eta$', color='g')
  ax1.tick_params('y', colors='g') 
  # third plot ------------------------------------------------------------
  ax[1, 0].set_xlim([0, var.q.max()*3600])
  ax[1, 0].plot(var.q*3600, var.wc*100, '.', color='k')
  ax[1, 0].hlines([5, 7], 0, var.q.max()*3600, 
    linestyles=['--'], color='red')
  ax[1, 0].set_xlabel('$q$ [$\\mathrm{m^3/h}$]')
  ax[1, 0].set_ylabel('$wc$')
  # forth plot ------------------------------------------------------------
  coeff = np.polyfit(var.q, var.eta_h,3)
  curve_eff = np.poly1d(coeff)
  ax[1, 1].plot(var.q*3600, var.eta_h, '.', color='k')
  ax[1, 1].plot(q_test*3600, curve_eff(q_test), '--', color='gray')
  ax[1, 1].set_ylim(0, var.eta_h.max()*1.1)
  ax[1, 1].set_xlim(0, var.q.max()*3600)
  ax[1, 1].set_xlabel('$q$ [$\\mathrm{m^3/h}$]')
  ax[1, 1].set_ylabel('$\\eta$')
  # _______________________________________________________________________
  ax3 = ax[1, 1].twinx()
  mu_ref = var.mu.mean()*1000-1
  bins = np.linspace(mu_ref-10, mu_ref+10, 11)
  ax3.set_ylim(bins[0], bins[-1])
  ax3.plot(var.q*3600, var.mu*1000, '.', color='g')
  ax3.hlines(var.mu.mean()*1000, 0, 100, color='g', linestyle='--')
  ax3.set_ylabel('$\\mu$ [cP]', color='g')
  ax3.tick_params('y', colors='g') 
  # _______________________________________________________________________
  ax4 = ax3.twiny()
  ax4 = sns.histplot(y=(var.mu*1000), bins=bins, alpha=0.3, stat='percent',
    color='r')
  ax4.set_xlim(0, 100)
  ax4.tick_params('x', colors='r') 
  # _______________________________________________________________________
  q_bep = q_test[np.argmax(curve_eff(q_test))]
  COEF.append(coeff)
  BEP_data.append([var.mu.mean(), var.w.mean(), q_bep, phi_bep])
  fig.suptitle((i, curves[i]))
  fig.tight_layout()
COEF = np.array(COEF)
BEP_data = np.array(BEP_data)

# %% =======================================================================
# analyses choke
# ==========================================================================
cases = {  'water':[0,   1,  2,  3, 4 , 5,  6, 7] , 
          '20 [c]':[8 , 11, 14, 17] , 
          '35 [c]':[9 , 12, 15, 18] , 
          '50 [c]':[10, 13, 16, 19] }
for i in cases:
  case = cases[i]
  df_sel = df_mean[np.isin(df_mean.i_curve, case)]
  var = variables(df_sel)
  cv = var.q*var.rho/(var.dp_ch*var.rho)**0.5
  plt.plot(var.ch, cv, ".", label=i)
plt.legend()
plt.xlabel('choke openig')
plt.ylabel('CV')


# %% =======================================================================
# analyses heat exchanger
# ==========================================================================
cases = {  'water':[4 , 5,  6, 7] , 
          '20 [c]':[8 , 11, 14, 17] , 
          '35 [c]':[9 , 12, 15, 18] , 
          '50 [c]':[13, 16, 19] }
for i in cases:
  case = cases[i]
  df_sel = df_mean[np.isin(df_mean.i_curve, case)]
  var = variables(df_sel)
  plt.plot(var.q, 1e-5*var.dp_he, '.', label=i)
plt.legend()
plt.xlabel('choke openig')
plt.ylabel('$\\Delta p$ heat exchanger [bar], ')
plt.show()
for i in cases:
  case = cases[i]
  df_sel = df_mean[np.isin(df_mean.i_curve, case)]
  var = variables(df_sel)
  plt.plot(var.q, 1e-5*(var.p0 - var.dp_he - var.p1) , '.', label=i)
plt.legend()
plt.xlabel('choke openig')
plt.ylabel('$\\Delta p$ tube [bar], ')


# %% =======================================================================
# analyses booster oil
# ==========================================================================
value = [160, 200, 240, 310, 350, 390, 430, 480, 520, 550, 630, 
         700, 760, 915]
df_an = pd.concat([df_w1, df_o, df_o1], axis=0, ignore_index=True)
for i in value:
  df_sel = df_mean[df_mean.booster_oil==i]
  var = variables(df_sel)
  fig, ax = plt.subplots()
  cb = ax.scatter(var.q*3600, var.p0*1e-5, c=var.mu*1000,
    vmin=1, vmax=180, cmap='tab10')
  ax.set_ylabel('$\\Delta p$ booster [bar], ')
  ax.set_xlabel('$q$ [$\\mathrm{m^3/h}$]')
  cb1 = plt.colorbar(cb)
  cb1.set_label('$\\mu$ [cP]')
  ax.set_title('$\\omega$ booster = %.0f'%(i))

# %% =======================================================================
# first view of bep
# ==========================================================================
mu =  BEP_data[:20,0]
w =   BEP_data[:20,1]
q =   BEP_data[:20,2]
phi = BEP_data[:20,3]
x =  np.linspace(200, 370)
colors = ['r', 'g', 'b']
i = [i for i in range(8)] # water
j = [10, 13, 16, 19]      # 50 C
k = [9, 12, 15, 18]       # 35 C
l = []
for en, index in enumerate([i,j,k]):
  plt.plot(w[index]*30/np.pi, q[index]*3600, 'x', color=colors[en])
  coeff = np.polyfit(w[index], q[index], 1)
  plt.plot(x, coeff[0]*x + coeff[1], '--', color=colors[en])
  print(coeff[0], coeff)
plt.ylabel('$q_{bep}$  measure   [$\\mathrm{m^3/h}$]')
plt.xlabel('$\\omega$ [rpm]')
plt.show()
plt.plot(mu[[0, 4, 8, 9, 10]]*1000, q[[0, 4, 8, 9, 10]]*3600, '.')
plt.plot(mu[[1, 5, 11, 12, 13]]*1000, q[[1, 5, 11, 12, 13]]*3600, '.')
plt.ylabel('$q_{bep}$  measure   [$\\mathrm{m^3/h}$]')
plt.xlabel('$\\mu$ [cP]')


# %% =======================================================================
# estimate bep
# ==========================================================================
def BEP(w, mu, coeff):
  return w*(coeff[0] + coeff[1]*mu #+ coeff[2]*mu**2
  ) + coeff[2] + coeff[3]*mu + coeff[4]*mu**2
def obj(coef):
  bep_pred = BEP(w, mu, coef) 
  return np.sum(( q - bep_pred)**2)
jac = jacobian(obj);  hes = hessian(obj)
jac = jacobian(obj);  hes = hessian(obj)
res = minimize(obj, [0.1, 0.1, 0.1, 0.1, 0.1], 
  jac=jac, hess=hes, method='trust-exact')
coef = res.x
bep_pred = BEP(w, mu, coef)
print('residual  =', res.fun, 'sucess = ', res.success)
print('coef = ', coef)
print('mae = ', 100*mean_absolute_error(q, bep_pred))
print('r2 = ',r2_score(q, bep_pred))
plt.plot(np.sort(q*3600), np.sort(q*3600), '--', 
  color='r', label='data')
plt.plot(np.sort(q*3600), np.sort(q*3600)+1, ':', 
  color='gray', label='1 [$\\mathrm{m^3/h}$] error')
plt.plot(np.sort(q*3600), np.sort(q*3600)-1, '--',
 color='gray')
plt.plot(q*3600, bep_pred*3600, '.', 
  color='k', label='pred')
plt.xlabel('$q_{bep}$  measure   [$\\mathrm{m^3/h}$]')
plt.ylabel('$q_{bep}$  predicted [$\\mathrm{m^3/h}$]')
plt.legend(); plt.show()

# %% =======================================================================
# save model
# ==========================================================================
path = os.path.join('result', 'model2.h5') 
model  = {} if len(glob.glob(path))==0 else h5todict(path)
model['bep'] = coef
dicttoh5(model, path, create_dataset_args=d_s_a)

# %% =======================================================================
# plot bep and get table
# ==========================================================================
q_pred = 3600*BEP(w, mu, coef)
q_teo = q*3600
dq = np.abs(q_pred- q_teo)
error = np.abs(dq)*100/q_teo
i = np.argsort(dq)
#---------------------------------------------------------------------------
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.step(dq[i], '--', color='k')
ax1.step(error[i], '--', color='r')
ax.set_xlabel('data')
ax.set_ylabel('$dq$ [$\\mathrm{m^3/h}$]')
ax1.set_ylabel('error %', color='r')
ax1.tick_params('y', colors='r')
columns = ['t', 2000, 2500, 3000, 3500]
df_q = pd.DataFrame(columns=columns)
df_m = pd.DataFrame(columns=columns)
df_q.t = np.array([25, 35, 50])
df_m.t = np.array([25, 35, 50])
for i in columns[1:]:
  df_q[i] = BEP(
    np.ones_like(df_q.t)*i*np.pi/30, 
    MU(df_q.t), coef)*3600
  df_m[i] = df_q[i]*RHO(df_q.t)
print(np.round(df_q))
var = variables(df_o)
CV_b = np.polyfit(var.w_b*30/np.pi, var.q*3600, 1)
CV_b[0]*100 + CV_b[1]

# %% =======================================================================
# plot bep by case
# ==========================================================================
#cases = [[0, 4], [1, 5], [2, 6], [3, 7]]; title = 'water'
#cases = [[0], [10], [9], [8]];            title = '2000 [rpm]' 
#cases = [[1], [11], [12], [13]];          title = '2500 [rpm]' 
#cases = [[2], [16], [15], [14]];          title = '3000 [rpm]' 
cases = [[3], [19], [18], [17]];          title = '3500 [rpm]'
colors = ['k', 'g', 'r', 'b']
fig, ax = plt.subplots()
for i, case in enumerate(cases):
  var = variables(df_mean[np.isin(df_mean.i_curve, case)])
  ax.plot(var.q*3600, var.eta_h*100, '.', color=colors[i], alpha=0.5)
  w_i = var.w.mean()
  mu_i = var.mu.mean()
  bep = BEP(w_i, mu_i, coef)
  if title=='water':
    label = '$\\omega$ [rpm]= %.0f'%(w_i*30/np.pi) 
  else:  
    label = '$\\mu$ = %.0f [cP]'%(mu_i*1e3)

  ax.vlines(bep*3600, 0, var.eta_h.max()*100, color=colors[i],
    linestyle='--', label=label)
ax.set_xlabel('$q [m^3/h]$')
ax.set_ylabel('$\\eta $')
ax.set_title(title)
ax.legend(bbox_to_anchor=(1.01, 0.6), ncol=1, loc='lower left')

# %%
