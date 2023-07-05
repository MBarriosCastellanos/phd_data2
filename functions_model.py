# %% =======================================================================
# Import libraries 
# ==========================================================================
import matplotlib.pyplot as plt                 # Plot several images
import numpy as np
import pandas as pd
from silx.io.dictdump import h5todict, dicttoh5
from functions import RE, df_SI
import os

# %% ========================================================================
# get model
# ===========================================================================
path = os.path.join('result', 'model2.h5') 
model = h5todict(path)

# %% ========================================================================
# fluid model
# ===========================================================================
coef = model['mu'] 
coef_rho = model['rho']
coef_bep = model['bep']
MU = lambda t: coef[0]*np.exp(-coef[1]*t**coef[2])
RHO = lambda t: coef_rho[0]*t + coef_rho[1]
def BEP(w, mu):
  return w*(coef_bep[0] + coef_bep[1]*mu #+ coef_bep[2]*mu**2
  ) + coef_bep[2] + coef_bep[3]*mu + coef_bep[4]*mu**2

#%% ===========================================================================
# steady state modeling
# =============================================================================
# Inlet pipe and reservoir ....................................................
def DP_in(q, mu, rho):
  d = 93e-3;                        # pipe diameter [m]
  rug = 0.26/1000                   # cast iron
  leq = model['booster'][-1]
  Re = RE(rho,d,q,mu)
  f = (64/Re)*(Re<2300) + (
    (1/ (-1.8*np.log10( ((rug/d)/3.7)**1.11 + (6.9/Re) )))**2 )*(Re>=2300)
  V = 4*q/(np.pi*d**2)
  return rho*f*(leq/d)*(V**2/2)
def Reservoir(rho, mu, w_b, q=3e-4, p_in=1e5):
  cv, m, b, _ = model['booster']
  dp_booster = DP_in(q, mu, rho) + p_in 
  return cv*w_b - dp_booster*(w_b*(m*1e-10 /mu) + b*1e-10 )
# ESP and Output pipe ........................................................
def DP_out(q, mu, rho, only_oil=True):
  d = 93e-3;              # pipe diameter [m]
  rug = 0.26/1000         # cast iron
  l_pipe = 1.128 + 0.67 + 1.645 + 7.62 + 16 + 2.6 #pipe from p9 to selection valve
  l_codes = 4*30*d       # 4 codes 90
  l_valve = 3*d          # 1 ball valve full open 
  # 3 codes 90  1 valve full open 
  if only_oil==True:
    l_pipe = l_pipe + 3.2 + 0.74 + 0.28 + 1.48 + 0.3
    l_codes = l_codes + 30*d + 2*16*d   # 1 code 90 + 2 codes of 45
  else: 
    l_pipe = l_pipe + 1.28 + 0.5 + 1.37 + 0.23
    l_codes = l_codes + 3*30*d          # 3 of  90
  Re = RE(rho,d,q,mu)
  f = (64/Re)*(Re<2300) + ( # Haaland
    (1/ (-1.8*np.log10( ((rug/d)/3.7)**1.11 + (6.9/Re) )))**2 )*(Re>=2300)
  l = l_pipe + l_codes + l_valve
  V = 4*q/(np.pi*d**2)
  return rho*f*(l/d)*(V**2/2)
def sigmoid (x, mu, case=0):
  z = mu*4
  [x0, y0, m1, m2] = model['choke'] if case==0 else model['choke_1']
  xp = m1*(x - x0)
  y = 1/(1 + np.exp(-xp))
  cv = m2*(y + y0) if case==0 else m2*(y + y0/(z**0.2)) 
  return cv/1000
def ESP_ss(rho, mu, q, w, z):
  # output pipe ..............................................................
  dp_out = DP_out(q, mu, rho)
  cv = sigmoid(z, mu)
  dp_choke = (q**2)*rho/(cv**2)
  p_out = dp_out + dp_choke 
  # ESP ......................................................................
  c1, c2, c4, k4, k5, n = model['esp']
  Re_inv = mu/rho/q
  K = k4*Re_inv + k5*Re_inv**n
  h = 1e-3*c1*w**2 - 1e1*c2*q*w - (K*1e7 + c4*1e6)*q**2 
  #p_in = p_out - h*rho*9.81
  #return h, p_in
  return h, p_out
# System .....................................................................
def System_ss(rho, mu, w_b, w, z):
  q = Reservoir(rho, mu, w_b)
  p_in = 1e5
  error = 100
  while error>0.01:
    q_bef = q; 
    p_in_bef = p_in
    h, p_out = ESP_ss(rho, mu, q, w, z)
    p_in = p_out - h*rho*9.81
    q = Reservoir(rho, mu, w_b, q, p_in)
    error1 = np.abs(np.sum((q    - q_bef   )/q   ))
    error2 = np.abs(np.sum((p_in - p_in_bef)/p_in))
    error = error1*100 + error2*100
    print("error = %.4f"%error)
  return h, p_out, q

#%% ===========================================================================
# dynamical modeling
# =============================================================================
# dynamical system ..........................................................
def dynamical(U, X):
  AB = model['dynamic']
  A = np.array([[AB[0], AB[1]*1e-4], [AB[2], AB[3]*1e-4]])
  B = np.array([[AB[4], AB[5]*1e-4], [AB[6], AB[7]*1e-4]])
  return A @ X + B @ U

#%% ===========================================================================
# Get variables from dataframe
# =============================================================================
def get_variables(df, dyn=False):
  # Fluid properties 
  #----------------------------------------------------------------------------
  rho = df.Density_oil;   mu = MU(df.T_2)*1e-3  # density, viscosity

  # Control variables
  #---------------------------------------------------------------------------- 
  w = df.ESP_rotation;    z = df.choke_esp;   w_b = df.booster_oil 

  # Pressure and flow
  #---------------------------------------------------------------------------- 
  p_in = df.P_1;          p_out = df.P_9;     h = (p_out - p_in)/rho/9.81 
  q = df.Q_oil/rho
  variables = [rho, mu, w, z, w_b, p_in, p_out, h, q]
  # Gradient
  #----------------------------------------------------------------------------
  if dyn:
    variables.extend([np.array(df.dh), 
      np.array(df.dp_in), np.array(df.dp_out)]) 

  return variables