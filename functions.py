# %% =======================================================================
# Import libraries 
# ========================================================================== 
import numpy as np                      # mathematics
from nptdms import TdmsFile             # import .tdms files
import csv                              # import .csv  files
from scipy.interpolate import interp1d  # linear interpolation
import glob                                     #  inspect folders
import pandas as pd                             # mathematics
import matplotlib.pyplot as plt                 # Plot several images
from scipy.fft import   fft, fftfreq, ifft      # function fo fast fourier 
from scipy import signal                        # signal statiscis
import os                               # manipulate folders
import time                                     # time calculation
sep = os.path.sep                       # path separator independin of OS

# %% =======================================================================
# Import libraries 
# ==========================================================================
def df_SI(dfi):
  # transform dataframe in international system
  df = dfi.copy()
  df.DP_emulsion*=100           # mbar => Pa
  df.DP_oil*=100                # mbar => Pa
  df.DP_he*=1e5                 # bar => Pa
  df.DP_choke*=1e5              # bar => Pa
  for i in ['P_' + str(i) for i in range(0,10)]:
    df[i]*=1e5                  # bar => Pa
  df.ESP_rotation*=2*np.pi/60   # rpm => rad/s
  df.Q_oil/=3600                # kg/h => kg/s
  df.Q_water/=3600              # kg/h => kg/s
  df.Q_water_tank/=3600         # kg/h => kg/s
  df.Watercut/=100              # 0 - 100 => 0 - 1
  df.booster_oil*=2*np.pi/60    # rpm => rad/s
  df.booster_water*=2*np.pi/60  # rpm => rad/s
  df.choke_esp/=100             # 0 - 100 => 0 - 1
  df.esp_rotation*=2*np.pi/60   # rpm => rad/s
  df.ESP_power*=1000            # kW => W
  return df
RE = lambda rho, d, q, mu: 4*rho*q/(mu*np.pi*d)

# %% =======================================================================
# Classes and Functions
# =========================================================================
class Sample(object):
  def __init__(self, path): 
    #print('Get data of ' + path)
    #self.path_control_signal = path + sep + 'control_signals.csv'
    self.path_control_signal = os.sep.join(
      path.split(os.sep) + ['control_signals.csv'])
    self.path_process = path + sep + 'process.tdms'
    self.data_control_in = {}     # data of control original len
    self.data_control = {}        # data of control of size  self.len
    self.data_process = {}        # data of process
    self.data = {}                # total data
  def get_csv(self):
    #print('... getting control data')
    # set control variables in continuous files ============================
    if len(glob.glob(self.path_control_signal))>0:
      self.csv = open_csv(          # open a .csv  file and import
        self.path_control_signal)
      self.controls = sorted(       # names of control variables
        np.unique(self.csv[:,1]))           
    else:
      self.csv = np.empty((1,4))
      self.controls = ['booster_oil', 'booster_water', 'choke_booster_water',
        'choke_esp',  'esp_rotation']
      #print('WARNING:: there are not ' + 
      #  '==_.csv file of control variables')
  def get_tdms(self):
    # process ==============================================================
    #print('... getting process data')
    if len(glob.glob(self.path_process))>0:
      self.tdms = TdmsFile.read(    # open a .tdms file and import
        self.path_process) 
      self.names = sorted(          # names of process variables 
          list(self.tdms['channels']._channels.keys()))
      self.start_time = self.tdms[  # initial time
        'channels'][self.names[0]].properties['wf_start_time']
      self.len = len(               # len of a variable of process data
        self.tdms['channels'][self.names[0]][:])
    else:
      print('WARNING:: there are not process.csv file of control variables')
  def set_data_time(self):
    '''     
    Function to create a time column of data, relative to the initial
    time of experiment self.start_time, the variable is saved in
    self.data_process['time']
    '''
    #print('... setting time of experiment')
    if self.names.count('timestamp')>0: # have a timestamp column 
      self.data_process['time'] = np.array(self.tdms[
        'channels']['timestamp'][:] - self.start_time, dtype='float')*1e-6
    else:                               # don't have a timestamp columns
      print('There are note timestamp vector in original data')
      self.data_process['time'] = np.array(
        self.tdms['channels'][self.names[0]].time_track(), dtype='float')
  def set_data_process(self):
    '''
    Function to obtain the process data of .tdms file in
    self.data_process as a dict
    '''
    #print('... setting process data')
    for name in self.names:       # for in channel names
      self.data_process[name] = select_interpolate(
        x = self.tdms['channels'][name][:], 
        t = self.data_process['time'], 
        l = self.len)             # adjust data vector to time vector size
    if self.names.count('timestamp') > 0:   # delete timestamp key and data 
      self.timestamp = {'timestamp': [
        str(i) for i in self.data_process['timestamp']]}
      del self.data_process['timestamp']
    else:
      print('WARNING::there are not timestamp vector')
      self.timestamp = {'timestamp': str(self.start_time)}
  def set_data_control(self):
    '''
    Function to split the data control 'self.csv' file with the columns:\\
    [0] = timestamp where the control variable change\\
    [1] = new control variable\\
    [2] = last control variable\n
    in a dict 'self_control_in[c]', with the control key 'c'. Every key of
    dict have three columns of\\
    [0] = relative time vector\\
    [1] = new control variable\\
    [2] = last control variable\n
    self.controls are a list of unique control changed in experiments\n
    #=====================================================================\n
    Organize and fill the dict self.data_control[c] to the dict 
    self.data_control[c] is a dictionary with every key c,
    as a one of control variables of experiment, with the respective values
    of every time of variable self.data_process['time']
    '''
    print('... setting control data')
    time = np.array( np.array(self.csv[:,0],  # time relative in [s] 
        dtype='datetime64')     #   timestamp of control variable
        - self.start_time       #   initial time of experiment
        + np.timedelta64(3,'h'),#   3h because of schedule change US to BR 
      dtype='float')*1e-6       #   change from [micro-s] to [s] 
    values = np.array(          # values of control variables changes in
      self.csv[:,2:], dtype='float')  # respective timestamp
    if len(glob.glob(self.path_control_signal))>0:
      for c in self.controls:     # fill every key in dict with columns [1,2] 
        print(c)
        i = np.where(             #  in column [0] where the control is c 
          c == self.csv[:,1])[0]  #   
        print(np.shape(i))
        self.data_control_in[c] = np.c_[time[i], values[i]]
        print(np.shape(self.data_control_in[c]))
        # ================================================================== #
        self.data_control[c] = fill2(  t = self.data_process['time'], 
                                      t1 = self.data_control_in[c][:,0], 
                                      y1 = self.data_control_in[c][:,1],
                                      y_in = self.data_control_in[c][0,2])
    elif self.path_process.split(sep)[-1] == 'process.tdms':
      values = self.path_process.split(sep)[-2].split('_')
      for value in values:
        value = value.split('-')
        var = float(value[1])*np.ones(self.len)
        if value[0] == 'e':
          self.data_control['esp_rotation'] =  var
        elif value[0] == 'bo':
          self.data_control['booster_oil'] =   var
        elif value[0] == 'bw':
          self.data_control['booster_water'] = var
        elif value[0] == 'c':
          self.data_control['choke_esp'] =     var
      self.data_control['choke_booster_water'] = np.zeros(self.len)
  def set_data(self):
    '''
    Set all data of experiment in a dict self.data
    '''
    self.get_tdms()
    self.set_data_time()
    self.set_data_process()
    self.get_csv()
    self.set_data_control()

    self.data = {**self.data_process, **self.data_control}
    #print('... data structured as dict')

class Sample_vib(Sample):
  def __init__(self, path): 
    #print('Get data of ' + path)
    self.path_process = path + sep + 'vibration.tdms'
    self.path_control_signal = path + sep + 'not.csv'
    self.data_control_in = {}     # data of control original len
    self.data_control = {}        # data of control of size  self.len
    self.data_process = {}        # data of process
    self.data = {}                # total data

class Time_vib(object):
  '''
  array of data of dimension [m, n] where m samples and n time 
  '''
  def __init__(self, matrix, t=None):
    self.x = matrix 

#=========================================================================#
def select_interpolate(x, t, l):
  '''
  the variable x, with the respective time t, if the len(x) is different of
  the len l of the others variables of experiment, the new vector x1 of
  is len l is generated by linear interpolation.
  '''
  if l != len(x):
    t1 = np.linspace(t.min(), t.max(), num = len(x))
    f1 = interp1d(t1, x)
    return f1(t)
  else:
    return x 
#=========================================================================#
def open_csv(path): 
  '''
  open .csv file as an array, and replace ',' by '.'
  '''
  with open(path, newline='') as f:       # Open file of path
    reader = csv.reader(f, delimiter=';') #   
    data = np.array(list(reader))
  m, n = np.array(data).shape             # Shape array
  for i in range(m):                      # replace , by . 
    for j in range(n):
      data[i,j] = data[i,j].replace(',','.')
  return data
#=========================================================================#
def fill(t, t1, y1, y_in):
  '''
  len(t1)=len(y1) = m; len(t)-len(y) = n;  y[0] = y_in
  give the vector y1(t1) find the corresponding y(t), in other words, give 
  the corresponding possition of y1 in the vector t. After that, fill the
  vector y.
  '''
  t = np.array([t]).T;      t1 = np.array([t1])
  items = np.unique(np.where(t>t1, 1, 0).sum(1),return_index=True
    )[1][-len(t1.T):]           # determine where t1 is equal to t
  y = np.empty(t.shape[0])      # create a y vector of size n
  for i in range(len(items)-1): # fill the vector y with the corresponding
    y[ items[i] : items[i+1] ] = y1[i]  # y1[i], repeat y1[i] in after 
  else:                                 #   empty position of y[i]
    y[items[-1]:] = y1[-1]      # fill the end position and after
    y[:items[0]] = y_in         # fill the begin position and before
  return y
def fill2(t, t1, y1, y_in):
  '''
  len(t1)=len(y1) = m; len(t)-len(y) = n;  y[0] = y_in
  give the vector y1(t1) find the corresponding y(t), in other words, give 
  the corresponding possition of y1 in the vector t. After that, fill the
  vector y.
  '''
  df1 =  pd.DataFrame(t, columns=['time']) 
  df2 = pd.DataFrame(
    {'time':[df1['time'][0]], 'y': [y_in]}
  )
  df3 =  pd.DataFrame(np.c_[t1, y1], columns=['time', 'y'])
  df4 = pd.concat([df2, df3], ignore_index=True)
  df5 = pd.merge_asof(df1, df4, on='time', direction='backward')
  return np.array(df5['y'])
#=========================================================================#
def get_controls(paths,i,j):
  '''
  function to get the a dataframe of control from a list of foldernames
  cut ever path in a path[i:-j]
  '''
  dict_df  = {}         # get controls of paths
  for path in paths:    # for foldername in paths
    path = path[i:j]    # remove data/d_1ph .../ from foldername
    init = [            # determine position or '-' in path
      i for i in range(len(path)) if path[i] == '-']
    end = [             # determine position or '_' in path
      i for i in range(len(path)) if path[i] == '_']
    dict_df[path] = [   # get values of 'e', 'bo' and 'bw'
      int(path[init[i]+1:end[i]]) for i in range(3)]
    if len(init) == 5:  # if correspond a transient (tbw, tc or te)
      dict_df[path      # get value of 'c'
        ].append(float(path[init[3]+1:end[3]]))
      dict_df[path      # get value of 't' (transient) and 'type'
        ].extend([float(path[init[4]+1:]), path[end[3]+1:init[4]]])
    else:               # if correspond a static point
      dict_df[path      # get value of 'c'
        ].append(float(path[init[3]+1:]))
      dict_df[path      # get value of 't' (transient) and 'type'
        ].extend([0, 'not'])
  return pd.DataFrame.from_dict( # transform in a dataframe
    dict_df, orient='index',columns=['e','bo', 'bw', 'c', 't', 'type'])
#==========================================================================#
def compare_df_column(column, df, key='c'):  
  '''
  Compare a column of data with a key of dataframe df[key]
  '''
  key_unique = np.unique(df[key])   # unique choke values in foldernames
  col_unique = np.unique(column)    # unique choke values in column
  key_replace = []                  # values to replace
  key_range = []                    # range of values to replace
  options = []                      # options to any value of column
  for c in key_unique:              # determine possibles rounds
    option = col_unique[np.where(col_unique.round()==c)[0]]
    options.append(option)
    if len(option)>0:
      key_replace.append(((option.max()+option.min())*0.5).round(1))
      key_range.append((option.ptp()*0.5).round(1))
    else:
      key_replace.append(c)
      key_range.append(0.0)   
  return key_unique, key_replace, key_range, options
#=Cepstrum functions ======================================================#
def complex_cepstrum(x, n=None):
    r"""Compute the complex cepstrum of a real sequence.

    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.

    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.

    The complex cepstrum is given by

    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}

    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.

    See Also
    --------
    real_cepstrum: Compute the real cepstrum.
    inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.


    Examples
    --------
    In the following example we use the cepstrum to determine the fundamental
    frequency of a set of harmonics. There is a distinct peak at the quefrency
    corresponding to the fundamental frequency. To be more precise, the peak
    corresponds to the spacing between the harmonics.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import complex_cepstrum

    >>> duration = 5.0
    >>> fs = 8000.0
    >>> samples = int(fs*duration)
    >>> t = np.arange(samples) / fs

    >>> fundamental = 100.0
    >>> harmonics = np.arange(1, 30) * fundamental
    >>> signal = np.sin(2.0*np.pi*harmonics[:,None]*t).sum(axis=0)
    >>> ceps, _ = complex_cepstrum(signal)

    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(t, signal)
    >>> ax0.set_xlabel('time in seconds')
    >>> ax0.set_xlim(0.0, 0.05)
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t, ceps)
    >>> ax1.set_xlabel('quefrency in seconds')
    >>> ax1.set_xlim(0.005, 0.015)
    >>> ax1.set_ylim(-5., +10.)

    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum
    .. [2] M.P. Norton and D.G. Karczub, D.G.,
           "Fundamentals of Noise and Vibration Analysis for Engineers", 2003.
    .. [3] B. P. Bogert, M. J. R. Healy, and J. W. Tukey:
           "The Quefrency Analysis of Time Series for Echoes: Cepstrum, Pseudo
           Autocovariance, Cross-Cepstrum and Saphe Cracking".
           Proceedings of the Symposium on Time Series Analysis
           Chapter 15, 209-243. New York: Wiley, 1963.

    """

    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay

    spectrum = np.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay
def real_cepstrum(x, n=None):
    r"""Compute the real cepstrum of a real sequence.

    x : ndarray
        Real sequence to compute real cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.

    Returns
    -------
    ceps: ndarray
        The real cepstrum.

    The real cepstrum is given by

    .. math:: c[n] = F^{-1}\left{\log_{10}{\left|F{x[n]}\right|}\right}

    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform. Note that
    contrary to the complex cepstrum the magnitude is taken of the spectrum.


    See Also
    --------
    complex_cepstrum: Compute the complex cepstrum of a real sequence.
    inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.

    Examples
    --------
    >>> from scipy.signal import real_cepstrum


    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum

    """
    spectrum = np.fft.fft(x, n=n, axis=1)

    return np.fft.ifft(np.log(np.abs(spectrum)), axis=1).real
def inverse_complex_cepstrum(ceps, ndelay):
    r"""Compute the inverse complex cepstrum of a real sequence.

    ceps : ndarray
        Real sequence to compute inverse complex cepstrum of.
    ndelay: int
        The amount of samples of circular delay added to `x`.

    Returns
    -------
    x : ndarray
        The inverse complex cepstrum of the real sequence `ceps`.

    The inverse complex cepstrum is given by

    .. math:: x[n] = F^{-1}\left{\exp(F(c[n]))\right}

    where :math:`c_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.

    See Also
    --------
    complex_cepstrum: Compute the complex cepstrum of a real sequence.
    real_cepstrum: Compute the real cepstrum of a real sequence.

    Examples
    --------
    Taking the complex cepstrum and then the inverse complex cepstrum results
    in the original sequence.

    >>> import numpy as np
    >>> from scipy.signal import inverse_complex_cepstrum
    >>> x = np.arange(10)
    >>> ceps, ndelay = complex_cepstrum(x)
    >>> y = inverse_complex_cepstrum(ceps, ndelay)
    >>> print(x)
    >>> print(y)

    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum

    """

    def _wrap(phase, ndelay):
        ndelay = np.array(ndelay)
        samples = phase.shape[-1]
        center = (samples + 1) // 2
        wrapped = phase + np.pi * ndelay[..., None] * np.arange(samples) / center
        return wrapped

    log_spectrum = np.fft.fft(ceps)
    spectrum = np.exp(log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay))
    x = np.fft.ifft(spectrum).real
    return x
def minimum_phase(x, n=None):
    r"""Compute the minimum phase reconstruction of a real sequence.

    x : ndarray
        Real sequence to compute the minimum phase reconstruction of.
    n : {None, int}, optional
        Length of the Fourier transform.

    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.

    Returns
    -------
    m : ndarray
        The minimum phase reconstruction of the real sequence `x`.

    See Also
    --------
    real_cepstrum: Compute the real cepstrum.

    Examples
    --------
    >>> from scipy.signal import minimum_phase


    References
    ----------
    .. [1] Soo-Chang Pei, Huei-Shan Lin. Minimum-Phase FIR Filter Design Using
           Real Cepstrum. IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS-II:
           EXPRESS BRIEFS, VOL. 53, NO. 10, OCTOBER 2006

    """
    if n is None:
        n = len(x)
    ceps = real_cepstrum(x, n=n)
    odd = n % 2
    window = np.concatenate(([1.0], 2.0 * np.ones((n + odd) / 2 - 1), np.ones(1 - odd), np.zeros((n + odd) / 2 - 1)))

    m = np.fft.ifft(np.exp(np.fft.fft(window * ceps))).real

    return m
#==sigmoid==================================================================#
sigmoid = lambda a, x: a[0] + a[1]/(1 + np.exp(-a[2]*(x - a[3])))
def dsigmoid (a, x): 
  '''
  # first derivate of sigmoid function
  import sympy as sym
  a0 , a1, a2, a3,  x = sym.symbols('a_0, a_1,  a_2, a_3, x')
  y = a0 + a1/(1 + sym.exp(-a2*(x-a3)))
  sym.diff(y,x)
  '''
  Exp = np.exp(-a[2]*(x-a[3]))
  return a[1]*a[2]*Exp/(1 + Exp)**2 #
def plot_sigmoid(x, a, wc, h, l1, l2, x_max):
  '''
  Function for plot, sigmoid, derivate sigmoid, boundaries and intersection
  '''
  cs = [(0,0,0), (0.3,0.3,0.3), (0.7,0.7,0.7)]
  lb = a[0];    ub = a[0] + a[1]  # boundaries
  fig, ax = plt.subplots(1, 1)    # create curve
  y_max = sigmoid(a, x_max)       # y of maximum dy
  # line -----------------------------------------------------------------
  m = dsigmoid(a, x_max)          # slope 
  b = y_max - m*x_max             # y intersection
  yplot = np.linspace(lb, ub)     # linspace between boundaries
  # plot sigmoid function-------------------------------------------------
  ax.plot(x, sigmoid(a, x), '-',  color=cs[1], label='$h$ sigmoid')
  ax.plot(wc[2:-1], h[2:-1], 'o', color=cs[1], label='$h$ laboratory') 
  ax.set_xlabel('$wc$ [%]');      ax.set_xlim([0,100])
  ax.set_ylabel('$h$ [m]', color=cs[1])      # lab data
  # plot first derivate of sigmoid function -----------------------------
  ax2 = ax.twinx()                            # duplicate axis
  ax2.plot(x, dsigmoid(a, x), '--', color=cs[0], label='$h\'$')
  ax2.set_ylim([0, dsigmoid(a, x).max()])     # plot derivative
  ax2.set_ylabel('$h \'$', color=cs[0])        # y label
  # plot boundaries -----------------------------------------------------
  ax.vlines([wc[l1], wc[l2]] , lb, ub, linestyles=':', color=cs[2])  
  ax.hlines([lb, ub] , 0, 100,         linestyles=':', color=cs[2])  
  ax.plot(x_max, y_max, 'x', color=cs[2])          # center
  ax.plot( (yplot-b)/m, yplot , ':', color=cs[2])  # line
  fig.legend(bbox_to_anchor=(0.9, 0.3))
#==fast fourier transform ==================================================#
def dft(x, fs, n, type='amp'):
  '''having the signal in frequency domain wit 
  x = amplitude vector or matrix
  fs = sampling frequency 
  n = len(x)'''
  fourier = fft(x, axis=1)[:, 5:n//2]
  if type=='phase':         # phase fourier transform
    angles = np.angle(fourier, deg=True)
    xf = (360 + angles)*(angles<0) + angles*(angles>=0)
  else:                     # amplitude fourier transform
    xf = 2.0/n*np.abs(fourier) 
  f = fftfreq(n, 1/fs)[5:n//2]
  return f, xf
def filter_rpm(x):
  ''' Filter to rpm signal, in frequency test'''
  x_fft = fft(x)
  l = len(x)
  #x_fft = np.array([0 if i<0 else i for i in x_fft])
  y = 2.0/l*np.abs(x_fft[10:l//2])
  peak = signal.find_peaks(y, prominence=1)[0]
  fr = np.bincount(peak[1:]-peak[:-1]).argmax()
  n = round(len(x_fft)/fr)
  N = np.linspace(fr, n*fr-fr, num=int(n-1), dtype=int)
  for i in N:
   #x_fft[i+1] = (x_fft[i-2] + x_fft[i+2])*(-0.5)
   x_fft[i] = (x_fft[i-2] + x_fft[i+2])*(0.5)
   #x_fft[i-1] = (x_fft[i-2] + x_fft[i+2])*(-0.5)
  return abs(ifft(x_fft))
def filter_rpm_2(x):
  l = len(x)
  peaks = signal.find_peaks(-x, prominence=50)[0]
  for p in peaks:
    x[p] = (x[p-1] + x[p+1])*(0.5)
  peaks = signal.find_peaks(x, prominence=50)[0]
  for p in peaks:
    x[p] = (x[p-1] + x[p+1])*(0.5)
  w = int(10)
  x = np.array(
    [x[:w].mean() if i<w else x[i-w:i].mean() for i in range(l) ])
  return x
def lowpass(t, x, f=10):
  '''filter frequencies below the frequency f [Hz] in the signal x with
  time t '''
  Ns = len(t)*0.5/(t.max()-t.min());    # Nyquist frequency
  Wn = f/Ns                             # filter par
  sb, sa = signal.butter(3, btype='lowpass', Wn =Wn)    # filter butter
  return signal.filtfilt(sb, sa, x)
def filterfreq(x, F0, fs=250, Q=2):
  '''filter frequency f0 [Hz] in the signal x with sampling frequency f0
  and quality factor Q
  F0 = list of frequencies to fiter'''
  for f0 in F0:  
    sb, sa = signal.iirnotch(f0, Q, fs)
    x = signal.filtfilt(sb, sa, x)
  return x
mean_movil = lambda x, w: np.array(
    [x[:w].mean() if i<w else x[i-w:i].mean() for i in range(len(x)) ])
def combined_filter(t, x, freq_low, freq,freq_max):
  x = lowpass(t, x, f=freq_low)
  return filterfreq(x, [i*freq for i in range(1, int(freq_max//freq)+1)])
#==progress function =======================================================#
def printProgressBar (iteration, total, prefix = '', suffix = '', 
  decimals = 3, length = 50, fill = '█', printEnd = '\r'):
    percent = ('{0:.' + str(decimals) + 'f}').format(
      100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
# viscosity ================================================================#
mu = lambda T: 0.00026436*T**4 - 0.04864*T**3 + 3.436*T**2 - 114.28*T + 1610.3
# multiply matrix M to vector v ============================================#
def mul(v,M):
  '''multiply a vector of len m or n wit a matrix M of shape m, n'''
  m, n = M.shape
  if len(v)==m:   # if the len vector is m, 
    return ( v.reshape(m,1) @ np.ones((1,n)) )*M
  elif len(v)==n: # if the len vector is n,
    return ( np.ones((m,1)) @ v.reshape(1,n) )*M
# Calculate Root Mean Square ===============================================#
RMS = lambda x: (np.sum(x**2, axis=1)/x.shape[1])**0.5
# Integration function for a X matrix and t vector time ====================#
intt = lambda X, t: mul(t[1:] - t[:-1],  np.cumsum(X, axis=1)[:,1:] )
# Integration function for a X matrix and t vector time ====================#
def create_folder(path):
  ''''check if folder path exist, if not exist create the folder'''
  if len(glob.glob(path))==0:
    os.mkdir(path);     print('creating ... ' +  path)
  else: print(path + ' folder not create, folder exist')
# calculate elapsed time ===================================================#
def time_elapsed(start):
  '''function to calculate the elapsed time , based on start point'''
  now = time.time();          t = now - start           # total time in sec
  h = int(t/3600);            m = int((t - h*3600)/60)  # hour, minutes
  s = int(t - h*3600 - m*60); ms = int((t - h*3600 - m*60 - s)*1e4) # seconds
  f = lambda num: str(num).rjust(2,'0')   # adjust leading zeros
  print('elapsed time %s:%s:%s:%s........................................'%( 
    f(h), f(m), f(s), str(ms).rjust(4,'0')))
# enumerate a vector print =================================================#
def enum_vec(vector):
  ''' function to organized print vector num'''
  for i, v in enumerate(vector):
    end = '.\n' if (i+1)%4==0 or (len(vector)-1)==i else ', ' 
    print('[%s] = %s'%(i, v), end=end)
# filter df by a threshold =================================================#
def filter_df(df, column, threshold, change=''):
  'filter a dataframe in function of a threshold'
  df_name = f'{df=}'.split('=')[0]
  if change=='>':
    df = df[ df[column] >  threshold ]          # filter by column
  elif change=='<':
    df = df[ df[column] <  threshold ]          # filter by column
  else:
    df = df[ df[column] != threshold ]          # filter by column
  #df = df.set_index(np.arange(len(df)))         # re arrange by index
  print('%s shape = %s after  filter by %s'%(df_name, df.shape, column))
  return df
# filter df by nan and inf =================================================#
def filter_nan(X):
  ''' function to delete dataframe columns with NaN or Inf values.'''
  print('features with    Inf and NaN = {} '.format(len(X.columns)))
  exclude = np.unique(np.where(np.isnan(X))[1]) # columns with  nan values
  for exc in [-np.inf, np.inf]:   # add columns with inf values
    exclude = np.unique(np.r_[exclude, np.unique(np.where(X==exc)[1]) ])
  X = X[np.delete(X.columns, exclude)]  # delete columns selected
  print('features without Inf and NaN = {} '.format(len(X.columns)))
  return X
# function to split data   =================================================#
def likelihood (y, values):
  ''' likelihood of every value of (values) in y numpy array'''
  s = len(y) if len(y)!=0 else 1  # len of y corrected by zero division
  return np.array([len(np.where(y==v)[0])/s for v in values])
def gini (y, classes, weights=np.empty((0))):
  '''considering the (y) np array, encouter the gini coefficient for 
  classes and the weights '''
  weights = np.ones(len(classes)) if len(weights)==0 else weights
  return 1 - np.sum((likelihood(y, classes)*weights)**2)
def logit(y, c):
  p = likelihood(y, c)
  return np.log(p/(1-p))
def impurity(x, y, threshold, weighted=False):
  '''y impurity divided in threshold based on gini criteria'''
  c, mj = np.unique(y, return_counts=True)    # classes
  w = np.sum(mj)/(len(mj)*mj) if weighted==True else np.ones(len(c)) #weights
  left = np.where(x<=threshold)[0];   right = np.where(x>threshold)[0]
  return np.sum([len(i) /len(y)*gini(y[i], c, w) for i in [left, right]])
# %%
