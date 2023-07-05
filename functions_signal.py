# %% =======================================================================
# Import dict
# ==========================================================================
from importlib.resources import path
from silx.io.dictdump import h5todict#, dicttoh5 # import, export h5 files
import numpy as np                              # mathematics
import pandas as pd                             # mathematics
import glob                                     #  inspect folders
from scipy import signal, stats                 # signal statiscis
from thermo.chemical import Chemical            # chemical properties
from copy import deepcopy                       # dict structures deepcopy
import matplotlib.pyplot as plt                 # Plot several images
from matplotlib import cm                       # cm
from scipy.interpolate.interpolate import interp1d
from functions import real_cepstrum, complex_cepstrum, mu, dft, mul, RMS
  # cepstrum,  viscosity function, function fo fast fourier transform   
  # multiply matrix to vector

#%% ========================================================================
# functions to get basic means of experiments
# ==========================================================================
# give name to previous boundaries------------------------------------------
get_var = lambda step, maxx: [str(int(i/100)) + '_' + 
    str(int(i/100+step/100)) + 'hHz' for i in np.arange(0,maxx,step)]
#---------------------------------------------------------------------------
def basic_inf(folders):
  ''' function to  find some basic information and data from every curve 
  experiment  for a set of folders '''
  dict_fold = {'path':[], 'curve':[], 'exp':[], 'point':[],'Q':[], 'dp':[],
    'wc':[], 'rho':[], 'mu':[],'T':[], 'w':[], 'Ta':[], 'p1':[], 'p9':[], 
    'i':[], 'dp_oil':[], 'dp_emulsion':[], 'rho_oil':[], 'power':[], 
    'q_oil':[], 'q_water':[], 'bo':[], 'bw':[], 'ch_bw':[], 'ch':[], 
    'n_curve':[], 'T2':[]}
  for f_i, folder in enumerate(folders):      # for in every folder
    paths = glob.glob(folder);  print(folder) # files in folder
    for p in enumerate(paths):                # file in files
      exp = h5todict(p[1]);                   # get process file
      path_vib = p[1].replace('process','vibration') # vibration path
      if len(glob.glob(path_vib))>0:          # if there are vibration
        # Water moved by pump water ----------------------------------------
        rho_w = Chemical('Water',T = exp['T_2'].mean()+ 273.15).rho   
        Q_win = exp['Q_water'].mean()/rho_w   # Q water moved by water pump
        # Oil moved by pump oil (included water)----------------------------
        Q_mix = exp['Q_oil'].mean()/exp['Density_oil'].mean()
        # Total Q_water and Q_oil ------------------------------------------
        Q_water = Q_mix*exp['Watercut'].mean()*0.01 + Q_win # Total Q water
        Q_oil = Q_mix*(1- exp['Watercut'].mean()*0.01)  # total Q oil
        WC = Q_water*100/(Q_oil + Q_water)              # Water cut
        DP = exp['P_9'].mean() - exp['P_1'].mean()      # pressure diff
        rho = rho_w*(Q_win/(Q_win + Q_mix)              # density all
          ) + exp['Density_oil'].mean()*(Q_mix/(Q_win + Q_mix))
        T = exp['ESP_torque'].mean()                    # Torque   
        w = exp['ESP_rotation'].mean()                  # Rotation speed
        Ta = T / (                              # Dimensionless Torque
           rho * 0.108**5 * w**2 * (2*np.pi/60)**2 )    
        path_split = p[1].replace('\\', '/').split('/') # split path
        dict_fold['path'].append(p[1])                  # path number
        dict_fold['curve'].append(path_split[1])        # curve 
        dict_fold['exp'].append(path_split[2])          # experiment
        dict_fold['point'].append(int(path_split[3][8:-3])) # exp point
        dict_fold['mu'].append(mu(exp['T_2'].mean()))   # viscosity
        dict_fold['Q'].append(Q_oil+ Q_water);   dict_fold['dp'].append(DP)
        dict_fold['wc'].append(WC);     dict_fold['rho'].append(rho)
        dict_fold['T'].append(T);       dict_fold['w'].append(w)
        dict_fold['Ta'].append(Ta)
        dict_fold['p1'].append(exp['P_1'].min())          
        dict_fold['p9'].append(exp['P_9'].mean())   
        dict_fold['i'].append(exp['ESP_current'].mean())# current
        dict_fold['dp_oil'].append(exp['DP_oil'].mean())# dp oil and emu
        dict_fold['dp_emulsion'].append(exp['DP_emulsion'].mean())
        dict_fold['rho_oil'].append(exp['Density_oil'].mean())#rho oil
        dict_fold['power' ].append(exp['ESP_power'].mean()) # power
        dict_fold['q_oil'].append(Q_oil)                # Q_oil total
        dict_fold['q_water'].append(Q_water)            # Q_water total
        dict_fold['bo'].append(exp['booster_oil'].mean())    
        dict_fold['bw'].append(exp['booster_water'].mean())
        if any(np.array(list(exp.keys()))=='choke_booster_water'):
          dict_fold['ch_bw'].append(exp['choke_booster_water'].mean())
        else:
          dict_fold['ch_bw'].append(0)
        dict_fold['ch'].append(exp['choke_esp'].mean())
        dict_fold['n_curve'].append(f_i)
        dict_fold['T2'].append(exp['T_2'].mean())

  return deepcopy(dict_fold)
class time_signal(object):
  ''' Class time signal'''
  def __init__(self, X, t, w, n_al) :   
    self.X = X; del X                               # matrix of time
    self.w = w; del w                               # frequency vector
    self.t = t; del t                               # time vector 
    self.n_al = n_al                                # pump blade numbers
    [self.m, self.n] = self.X.shape                 # size or X shape(m,n)
    win = signal.windows.hann(self.n, np.pi*0.5)        # windows hanning
    fs = len(self.t)/(self.t.max()-self.t.min())        # sampling frequency
    self.f, self.Y = dft( mul(win, self.X), fs, self.n) # fft in amplitude
    # welch transform ======================================================
    self.f_w, self.Y_w  = signal.welch(self.X, fs, window='hanning', axis=1) 
    # cut frequencies after 10 kHz =========================================
    if self.f.max()>10100:
      [self.f, self.Y] = get_y([self.f, self.Y], [10, 10100], axis='both') 
      [self.f_w, self.Y_w] = get_y([self.f_w, self.Y_w], [10, 10100], 
        axis='both')
    # fft/omega, fft/omega^2, fft/amp_omega, fft*f, fft*f/omega
  def signal(self, s):
    '''define kind of signal'''
    if s=='fftw':             # fft/omega
      return mul(1/self.w,  self.Y)           
    elif s=='fftw2':          # fft/omega^2
      return mul(1/self.w**2,  self.Y)        
    elif s=='ffta':           #  fft/amp_omega
      a = get_ij(self.f,  [self.w - 2,  self.w + 2], self.Y)
      return mul(1/a,  self.Y)
    elif s=='fftn':           # fft*/amp_n_blades
      a_n = get_ij(self.f,    
        [self.w*self.n_al-2, self.w*self.n_al+2], self.Y) 
      return mul(1/a_n,  self.Y)
    elif s=='fftf':           # fft*f
      return mul(self.f,  self.Y)
    elif s=='fftfw':          # fft*f/omega 
      return mul(1/self.w,  mul(self.f,  self.Y))
    elif s=='ceps1':          # cepstrum
      return get_ceps( [self.t, self.X]) 
  def sts(self, sig, bdy, norm_kind, stat, split=''):
    '''
    statiscis of each signal 
    sig =  list of signals
    bdy = frequency boundaries 
    norm_kind = normalization of signal
    split = split by experiment or complete matrix
    '''
    i = 0 if len(split)==0 else split[0]        # initial boundary
    j = self.m if len(split)==0 else split[1]   # final boundary 
    for s in sig:                               # for in each signal
      print(f'\r{s}', end='')
      if s=='time':       # time signal
        x = get_x( [self.t, self.X[i:j]], bdy);         f = ''
      elif s=='ceps1':    # cepstrum signal
        x = get_ceps( [self.t, self.X[i:j]], bdy=bdy);  f = ''
      elif s=='welch':    # welch signal
        [f, x] = get_y( [self.f_w, self.Y_w[i:j]] , bdy, axis='both')
      elif s=='fft':      # fft frequency signal
        [f, x] = get_y( [self.f, self.Y[i:j]] , bdy, axis='both')
      else:           #  'fftw', 'fftw2', 'ffta', 'fftn', 'fftf', 'fftfw'
        [f, x] = get_y( [self.f, self.signal(s)[i:j]] , bdy, axis='both')
      # After get signal is normalized =====================================
      x = Norm(x, kind=norm_kind)     
      # After normalization the statistics is find =========================
      col = sts(stat, x, f=f) if sig[0]==s else np.c_[col, sts(stat, x, f=f)]
    return  col # getting column of statistics
def sts(stat, x, f=''):
  '''
  for a vector x find the statistics
  stat =  [ 'mean', 'var', 'skw', 'kur', 'ent', 'kstat1', 'vari', 'iqr',  
    'sem', 'bstd', 'gmean', 'hmean', 'gstd', 'rms', 'crest', 'wmean',
    'trapz']
  '''
  col = np.empty((x.shape[0],0))
  x[x==0]  = 1e-308                 # avoid division zero and inf errors
  if any([s=='mean' or s=='var' or s=='skw' or s=='kur' for s in stat]):
    a = stats.describe(x, axis=1)
  if any([s=='hmean' or s=='wmean' for s in stat]):
    hmean = x.shape[1]/np.sum(1/x, axis=1)          # harmonic mean
  if any([s=='rms' or s=='crest' for s in stat]):
    rms = RMS(x)                                    # root mean square 
  if any([s=='crest' for s in stat]):
    x_max = x.max(axis=1);    x_min = x.min(axis=1)
    peak1 = np.abs(x_max )              *(x_min>=0) # peak middle amplitude
    peak2 = (np.abs(x_max  - x_min)*0.5)*(x_min<0)  # peak all amplitude
    peak  =  peak1 + peak2 # Select peak1 or peak2, if have negative value
  for s in stat:
    if s=='mean':
      col = np.c_[col, a.mean] 
    elif s=='var':
      col = np.c_[col, a.variance]
    elif s=='skw':
      col = np.c_[col, a.skewness]
    elif s=='kur':
      col = np.c_[col, a.kurtosis]
    elif s=='hmean':
      col = np.c_[col, hmean ]
    elif s=='wmean':
      col = np.c_[col, np.sum(f)/( 
        f.reshape(1, len(f))  @ (1/x).T  )[0] if len(f)>0 else hmean ]
    elif s=='rms':
      col = np.c_[col, rms]
    elif s=='crest':
      col = np.c_[col, peak/rms]
    elif s=='ent':
      col = np.c_[col, stats.entropy(x, axis=1)]
    elif s=='kstat1':
      col = np.c_[col, np.array(
        [stats.kstat(x[i], n=1)  for i in  range(x.shape[0]) ])]
    elif s=='vari':
      col = np.c_[col, stats.variation(x, axis=1)]
    elif s=='iqr':
      col = np.c_[col, stats.iqr(x, axis=1) ]
    elif s=='sem':
      col = np.c_[col, stats.sem(x, axis=1) ]
    elif s=='bstd':
      col = np.c_[col, np.array([ # bayes mean
        stats.bayes_mvs(x[i])[2][0] for i in  range(x.shape[0])]) ]
    elif s=='gmean':
      col = np.c_[col, stats.gmean(np.abs(x), axis=1) ]
    elif s=='gstd':
      col = np.c_[col, stats.gstd( np.abs(x), axis=1) ]
    elif s=='trapz':
      col = np.c_[col, np.trapz(x, axis=1) ]
  return col

#%% ========================================================================
# functions to get basic means of experiments
# ==========================================================================
def get_y(fy, bdy, axis='y'):
  '''function to cut a matrix fy[1] and a vector fy[0] in the position bdy
  fy[0]= frequency vector, fy[1] = matrix amplitude, bdy = [i1, i2]
  '''
  if len(bdy)>0:
    [i, j] = [np.where(fy[0]<=bdy[0])[0][-1], np.where(fy[0]>=bdy[1])[0][0]]
    fy = [fy[0][i:j + 1], fy[1][:, i:j + 1]]
  return fy[1] if axis=='y' else fy
def get_x(tx, l, btype='bandpass', axis='y'):
  '''
  getting a vector t = tx[0] and a amplitude matrix x = tx[1], and 
  frequency limits l = [i1, i2], make o filter of time signal taking it 
  to frequency domain, cutting in frequency limits i1-i2 and come back to 
  time domain
  '''
  if len(l)>0:
    Ns = len(tx[0])*0.5/(tx[0].max()-tx[0].min());    # Nyquist frequency
    Wn = [l[0]/Ns, l[1]/Ns] if btype=='bandpass' else l/Ns  # filter par
    sb, sa = signal.butter(3, btype=btype, Wn =Wn)    # filter butter
    tx = [tx[0], signal.filtfilt(sb, sa, tx[1])]      # digital filter 
  return tx[1] if axis=='y' else tx
def get_ceps(time, bdy='', kind='real'): 
  ' function to get ceps of a signal in time domain'
  if kind=='real': 
    time[1] = np.abs(real_cepstrum(   get_x(time, bdy))   )
  else:
    time[1] = np.abs(complex_cepstrum(get_x(time, bdy))[0])
  return get_y(time, [1, time[0].max()-1]) # cut between limits
def Norm(x, kind='x'):
  '''function to normalize the signal before found the statistics'''
  if kind =='log':      # give the signal to the logarithmic scale
    x = np.log10(np.abs(x))
  elif kind =='rms':    # signal divide by the RMS of the signal
    x = mul(1/RMS(x), x)  
  elif kind == 'norm':  # signal normalized between 0-1
    xmin = mul(x.min(axis=1), np.ones(x.shape))
    xmax = mul(x.max(axis=1), np.ones(x.shape))
    x = (x- xmin)/(xmax - xmin) 
  elif kind == 'stad':  # signal normalized between -3 and 3
    xmean = mul(x.mean(axis=1), np.ones(x.shape))
    xstd = mul(x.std(axis=1), np.ones(x.shape))
    x = (x-xmean)/xstd 
  elif kind == 'log_2': # square signal logarithmic
     x = np.log10(x**2)
  elif kind == 'x2':    # square signal 
    x = x**2
  return x
def Norm2(X, kind): 
  '''function to test different filters in result signal, very similar to
  Norm'''
  if kind=='log':
    X = np.log10(np.abs(X))
  elif kind=='sqrt':
    X =np.sqrt(np.abs(X))
  elif kind=='logsqrt':
    X = np.log10(np.sqrt(np.abs(X)))
  return X
def get_columns(sig, stat, var): 
  ''' given two strings ex:(sig, stat) and a vector (ex: var=['a', 'b'])
   get [sig_stat_a, sig_stat_b']'''
  ijk = [sig, stat, var]                            # possible combination
  r = np.where([type(r)==list for r in ijk])[0][0]  # where replace in ijk
  compare = ijk[r];       ijk[r] = '*'              # comparason vector
  return ['_'.join(ijk).replace('*', c) for c in compare]
def get_ij(f, lims, y):
  '''
  Get de max peak in a signal between some limits.
  y = amplitude matrix fo size m, n
  f = frequency vector of len n
  lims = [ f[i1], f[i2] ] 
  For every row of y, determine the maximum value between the position
  i1 and i2 and return a as a column vector of size (m, 1)  
  '''
  i1 = np.where(f.reshape( 1,len(f) ) <= (      # find position i1
    lims[0]).reshape( len(lims[0]),1 ), 1, 0);  i1 = i1.sum(axis=1)
  i2 = np.where(f.reshape( 1,len(f) ) >= (      # find position i2
    lims[1]).reshape( len(lims[1]),1 ), 0, 1);  i2 = i2.sum(axis=1)
  a = np.array(
        [y[i] [i1[i]:i2[i]+1].max() for i in range(y.shape[0]) ])
  return a  

def add_vib(X, DF, x, path, i, N):
  '''
  DF = Dataframe with all data
  function to add x vibration data to matrix X
  X = Matrix with all vibration to be create
  x = Vector with vibration of one experiment
  path = process path
  i = index of experiment deleted
  N = len expected for vibration
  also to check len of vibration vector, add to X or del in DF
  '''
  n = len(x)            # experiment vibration len
  if n==N:              # if vibration len is equal 
    X = np.c_[X, x]     # add to vib data           
  elif n>N:             # if vibration len is more
    X = np.c_[X, x[:N]] # cut vib data and add to variable
    print('%s position %s, len = %s cutting'%(path, i, n))
  else:                 # if vibration len is less
    print('%s position %s, len = %s deleting'%(path, i, n))
    DF = DF.drop(index=i) # delete file
  return X, DF 

#%% ========================================================================
# plots
# ==========================================================================
def plot_columns(DF, curves, columns, path='', plot_kind='lin', 
  xi='wc', styles=['-o', '-x', '-+', '-d', '-v', '-*', '-|', '--o', '--x',
   '--+', '--d', '--v', '--*','--|', ':o', ':x', ':+', 
   ':d', ':v', ':*', ':|']):
  '''
  function to plot and compare every statistical analysis for different
  frequency ranges.
  DF = dataframe with all data
  curves = list of every experiment curve
  columns = columns of df to comparate
  plot_kind = plot kind, (lin) linear or  (log) logarithmic
  xi = variable name in horizontal axis extracted from DF
  '''
  col = '_'.join(columns[1].split('_')[1:]) # column name without stat_var
  # save folder -----------------------------------------------------------
  savef = '/'.join([path, xi + '_' + col + '_col.png'])
  if len(glob.glob(savef))<1 or path=='':       # if plot not exist
    plt.clf; styles = 1000*styles
    fig, axs = plt.subplots(1, len(columns), figsize=(len(columns)*12,8)) 
    for c_i, curve in enumerate(curves):        # plot every curve          
      df = DF[DF['curve']==curve];  x = df[xi]  # reduced df one curve
      for c, column in enumerate(columns):      # search avery column       
        axs[c].set_title(column);         axs[c].set_xlabel(xi)
        y = np.log10(np.abs( df[column])
          ) if plot_kind=='log' else df[column]
        axs[c].plot(x, y, styles[c_i], label=curve)
    lgd = plt.legend(bbox_to_anchor=(1.05, 0.5), ncol=1, 
      loc='center left'); 
    if path=='':
      plt.show()
    else:             
      plt.savefig(savef, bbox_extra_artists=(lgd,), bbox_inches='tight')    
    fig.clf();  plt.close();  #print(savef);#plt.show()
def plot_curves(DF, curves, columns, path='', xi='wc', styles=['-o', '-x', 
  '-+', '-d', '-v', '-*', '-|', '--o', '--x', '--+', '--d', '--v', '--*',
   '--|', ':o', ':x', ':+', ':d', ':v', ':*', ':|']):
  '''
  function to plot and compare every statistical analysis for different
  frequency ranges.
  DF = dataframe with all data
  curves = list of every experiment curve
  columns = columns of df to comparate
  plot_kind = plot kind, (lin) linear or  (log) logarithmic
  xi = variable name in horizontal axis extracted from DF
  '''
  col = '_'.join(columns[1].split('_')[1:]) # column name without stat_var
  # save folder -----------------------------------------------------------
  label_c = lambda curve: str(float(curve[-6:-3])/10) + ' [$m^3/h$]'
  savef = '/'.join([path, xi + '_' + col + '_curv.png'])
  if len(glob.glob(savef))<1 or path=='':               # if plot not exist
    plt.clf;        l = len(curves)         # curves to plot 
    fig, axs = plt.subplots(1, l, figsize=(l*6,4), sharey=True)
    for c in range(l):                      # for every curve
      df = DF[DF['curve']==curves[c]]       # df one curve
      df.plot(x =xi, y = columns, ax= axs[c], color='C' + str(c),
        style=styles)                       # one plot for curve
      axs[c].set_title(label_c(curves[c]))
    if path=='':
      plt.show()
    else:
      plt.savefig(savef);      
    fig.clf();  plt.close()

#%% ========================================================================
# plots colormesh
# ==========================================================================
def get_z(DF, x, y, x_i='wc', y_i='curve', z_i='Ta'):
  '''Function to find the equivalent Z to plot in colormesh 
  DF = DataFrame with all data, 
  x = all possible values of x_i in DF, x_i name of DF
  y = vertical axis values in colormesh, y_i name in DF
  z_i = name of z variable in DF.
  '''
  Z = []
  for i in y:
    df = DF[DF[y_i]==i] # Data for one curve ["curve"]==curve
    f = interp1d(       # interpolation function for wc_total
      df[x_i], df[z_i], bounds_error=False) 
    # limits in witch the curve change from water-in-oil to oil-in-water
    i1 = df.index[np.where(df.index<=df['i1'])[0][-1]]
    i2 = df.index[np.where(df.index>=df['i2'])[0][0]]
    #i1 = int(df['i1'].min());             i2 = int(df['i2'].min())
    i1 = np.where(x==df[x_i][i1])[0][0];  i2 = np.where(x==df[x_i][i2])[0][0]
    z = f(x)            # Find values for experiment df as row of Z
    # Add limits of phase change ............................................
    z[i1-3:i1+3] = np.nan;                z[i2-3:i2+3] = np.nan 
    Z.append(z)         # Add row to matrix
  return np.array(Z)
def Colormesh(x, y, Z,  y_ticks, xlabel='wc [%]',ylabel='curves', zlabel='Ta'):
  """
  Function to plot colormesh function, 
  x, y, vector to mesh
  Z matrix of results,  y_ticks.
  """
  plt.yticks(np.arange(len(y_ticks)), labels=y_ticks)
  plt.xlabel(xlabel, fontsize=25);    plt.ylabel(ylabel, fontsize=25);  
  plt.title(zlabel, fontsize=25)
  X, Y = np.meshgrid(x, y)    # Plot colormesh
  cs = plt.pcolormesh(X, Y, Z,  cmap=cm.jet, shading='auto')
  plt.colorbar(cs);                 plt.tight_layout()


#%% ========================================================================
# plots for test
# ==========================================================================
poly = lambda x, C: np.array([c*x**i for i, c in enumerate(C)]).sum(axis=0)
def plot_lims(DF, PR, kind, parameter, deg=1, norm=0, title=''):
  ''' function to plot the approximate transition with x = wc, y = parameter
  segmented by class'''
  plt.rcParams["figure.figsize"] = (8,8); plt.title(title) # figure format
  X = Norm2(PR, kind)                 # filter by parameter scale
  Xy = pd.concat([DF , X], axis=1)    # complete dataframe X, parameters 
  wr, pr = return_lim(Xy, parameter)  # get points of transition
  A = np.polyfit(wr, pr, deg)[::-1]   # fit points to a polynomial regresion
  styles = ['o', '.', '_']; alphas = [0.5, 1, 1]  # format
  #for i, label in enumerate(['água en óleo', 'óleo en água', 'transição']):
  for i, label in enumerate(['water-in-oil', 'oil-in-water', 'transition']):
    x = Xy[Xy['class']==2-i]['w']/60        # watercut
    y = Xy[Xy['class']==2-i][parameter]     # parameter
    y = y if norm==0 else y*A[0]/poly(x, A) # get curv normalized by bound
    plt.plot(x, y, styles[i], label=label, alpha=alphas[i])#  color='dimgrey'
  limx = np.arange(wr.min()*0.95, wr.max()*1.05)  # get boundary curve x
  limy = poly(limx, A)                            # get boundary curve y
  if norm==0: # plot boundary curve original
    plt.plot(limx, limy, '--', label='boundary', color='black')
    #plt.plot(limx, limy, '--', label='limite', color='black')
    plt.plot(wr, pr, 'X', color='black', alpha=0.7)
  else:       # plot boundary curve adjusted to new data
    plt.hlines(A[0], wr.min() , wr.max(), linestyles='--', color='black' ,
     label='boundary')
  plt.xlabel('$\Omega$ [Hz]');    plt.ylabel('$x_{RMS}$')
  plt.legend( loc='center right', bbox_to_anchor=(1.35, 0.5));  plt.show()
  return A    # Return data 
def return_lim(df, y):
  ''' function to determine the the boundaries in what the  
  the phase transition occur for every rotational speed and parameter'''
  df_dict = {}  # split the df in three dfs, one for every rotational speed 
  for i, j in enumerate([1800, 2400, 3000]): # three revelant omega 
    df_dict[i] = df[df['w']>j-300]  [df['w']<j+300] #one df in dict
  w = []; p = []
  for i in df_dict:   # for every individual df of each velocity
    w.append( df_dict[i]['w'].mean() )  # have the mean velocity
    l = []                              # limits
    for j in [1, 2]:  # for the class 1=water-in-oil 2=oil-in-water
      l.append( df_dict[i] [ df_dict[i]['class']==j ][y].min()  )
      l.append( df_dict[i] [ df_dict[i]['class']==j ][y].max()  )
    l = np.sort(l) # sort the maximum, minimum of each class for w[i]
    p.append((l[1] + l[2])*0.5) # the middle limits indicate the transition
  return np.array(w)/60, np.array(p) # simply return a vector of div
def plot_tree(clf, List):
  '''plot a simplest tree in the tablet'''
  n_nodes = clf.tree_.node_count    # number of nodes
  children_left = clf.tree_.children_left   # population to left node
  children_right = clf.tree_.children_right # population to right node
  feature = clf.tree_.feature               # every analysed feature
  threshold = clf.tree_.threshold           # where the division occur

  node_depth = np.zeros(shape=n_nodes, dtype=np.int64) #
  is_leaves = np.zeros(shape=n_nodes, dtype=bool)
  stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
  while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
      stack.append((children_left[node_id], depth + 1))
      stack.append((children_right[node_id], depth + 1))
    else:
      is_leaves[node_id] = True

  print("The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes))
  for i in range(n_nodes):
    if is_leaves[i]:
      print("{space}node={node} is a leaf node.".format(
        space=node_depth[i] * "\t", node=i))
    else:
      print("{space}node={node} is a split node: "
        "go to node {left} if  {feature_name} <= {threshold} "
        "else to node {right}.".format(
        space=node_depth[i] * "\t",
        node=i,
        left=children_left[i],
        threshold=threshold[i],
        right=children_right[i],
        feature_name=List[feature[i]]))

