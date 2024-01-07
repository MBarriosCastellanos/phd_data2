# %% =======================================================================
# Import libraries 
# ========================================================================== 
from silx.io.dictdump import h5todict, dicttoh5 # import, export h5 files
import autograd.numpy as np                     # mathematics 
import pandas as pd                             # dataframes
import os                                       # sort files 
import glob                                     # global folers
from thermo.chemical import Chemical    # chemical properties
from scipy.constants import g as g      # gravitationa constant
sep = os.path.sep                       # path separator independin of OS
from functions import Sample, Sample_vib, printProgressBar
from functions import combined_filter, df_SI, time_elapsed
d_s_a = {                               # arguments transform in h5 file
  'compression':'gzip', 'shuffle':True, 'fletcher32':True }
import time;    START = time.time()             # time calculation
print('Basic Libraries imported')
from nptdms import TdmsFile             # import .tdms files
import matplotlib.pyplot as plt
#plt.rcParams['legend.fontsize'] = 18
#plt.rcParams['xtick.labelsize'] = 18
#plt.rcParams['ytick.labelsize'] = 18
#plt.rcParams['axes.titlesize'] = 18
#plt.rcParams['axes.labelsize'] = 18


# %% =======================================================================
# list curves
# ========================================================================== 
curves = sorted(glob.glob('data/ct_1ph*'))[:24]
curves_kind = []
k = 0
df_inf = pd.DataFrame(columns=['k', 'i', 'curve', 'point', 'curve_kind'])
timestamp = []
for i, curve in enumerate(curves):
  print(('%s %s %s'%(i, curve, k)).center(70, '.'))
  curve_list = curve.split(sep)[-1].split('_')
  curve_kind = '_'.join(curve_list[:3])
  curves_kind.append(curve_kind)
  for path in sorted(glob.glob(curve + '/*-*')):
    df_inf.loc[k, :] = [k, str(i), curve, 
      path.split(sep)[-1], curve_kind] 
    k+=1
    path_process = path + sep + 'process.tdms'
    tdms = TdmsFile.read(path_process) 
    names = names = sorted(list(tdms['channels']._channels.keys()))
    start_time = tdms['channels']['timestamp'][:][0]
    start_time = np.array(start_time, dtype='datetime64')
    start_time = start_time.astype(np.int64) / 10**6
    timestamp.append(start_time)
curves_kind_unique = np.sort(np.unique(curves_kind))
df_inf['time'] = timestamp

df_inf = df_inf.sort_values(by=['curve', 'time'])
df_inf['k'] = range(len(df_inf))
df_inf.index = range(len(df_inf))

print('k_max = ', k); k_max = k
print('get basic data inf')
time_elapsed(START)               # Time elapsed in the process


# %% =======================================================================
# get process curve
# ==========================================================================
'''They take 15 seconds for each measurement point. \
  Vibration sampling is done at a rate of 51200 samples per second, \
    while process sampling is done at a rate of 250 samples per second. \
      Due to the filtering process, two seconds are lost at the beginning \
        and two seconds are lost at the end. \
As a result, the new size is 3250 for process sampling and \
  563200 for vibration sampling, \
each representing an 11-second duration.'''
k = 0
paths_del = []; DF = []; b = []; f =[]
#for i in df_inf.index:
curves = list(np.unique(df_inf.curve))
column_names = ['time', 'process_time', 'dp_ref',  'p1', 'p9', 'T_in',  
  'w_b', 'w' ,  'z', 'rho', 'mu', 'w_new', 'z_new']
for i, curve in enumerate(curves):
  points = df_inf[df_inf.curve==curve].point
  curve_process_path = os.sep.join([curve, 'process.csv'])
  curve_process = pd.read_csv(curve_process_path, names= column_names) 
  curve_process['w'] = curve_process['w']*30/np.pi*30/np.pi
  curve_process['w'] = curve_process['w']*30/np.pi*30/np.pi
  curve_process['w_b'] = curve_process['w_b']*30/np.pi
  curve_process['p1'] = curve_process['p1']*1e-5
  curve_process['p9'] = curve_process['p9']*1e-5
  curve_process['dp_ref'] = curve_process['dp_ref']*1e-5
  for point in points:
    printProgressBar(k + 1, k_max, prefix = '%s/%s'%(k+1,k_max), )
    path = sep.join([curve, point])
    # check point 
    k_df = df_inf[(df_inf.curve==curve) & (df_inf.point==point)]
    k_df = k_df.iloc[0, 0]
    #if k_df != k:
    #  print(' not coincide k '.rleft(70, '█')) 

    # get process experiment -----------------------------------------------
    exp = Sample(path);                 exp.set_data()
    df = pd.DataFrame(exp.data)
    if len(point.split('-'))>5:
      df = set_control_var(exp, df)
    start_time = (
      exp.start_time - np.datetime64('1970-01-01T00:00:00')
      ) / np.timedelta64(1, 's')
    df['time'] = df['time'] + start_time

    # get vibration experiment ---------------------------------------------
    exp_vib = Sample_vib(path);     exp_vib.set_data()
    df_vib = pd.DataFrame(exp_vib.data)

    # filtering of rotation sampling ---------------------------------------
    df['ESP_rotation'] = combined_filter(df['time'], df['ESP_rotation'], 
      10, 10/3, 20)

    # check relation of process vibration data ----------------------------
    rel = len(df_vib)/len(df)
    if rel <202 or rel>205:
      print(path)
      print('relation vib/process', len(df_vib)/len(df))
      paths_del.append(path)

    # cut the sampling due to filtering ------------------------------------
    df = df.iloc[500:-500];     df_vib = df_vib.iloc[102400: -102400]
    df.index = range(len(df));  df_vib.index = range(len(df_vib))

    # discretize process dataframe -----------------------------------------
    l = len(df)//25                         
    df_i = pd.DataFrame(columns=df.columns)
    for j in range(l):
      df_j = pd.concat([df.loc[j*25   :j*25+12, :].mean().T,
                        df.loc[j*25+12:j*25+24, :].mean().T], axis=1).T
      df_i = pd.concat([df_i, df_j], axis=0, ignore_index=True)
  
    

    # some aditional definition -------------------------------------------
    curve_kind = curves_kind[i] # curve kind fo this exp
    df_i['i_curve'] = i
    df_i['k'] = k;        
    #df_i['curve_kind'] = 0     # curve kind is the kind of curve
    df_i['curve_kind'] = np.where(curves_kind_unique==curve_kind)[0][0]
    df_vib['k'] = k

    df_i_merge = pd.merge_asof(
      df_i, curve_process, on='time', direction='backward')
    print( 'df_i = %s, df_i_merge = %s'%(df_i.shape, df_i_merge.shape))
    df_i_merge = df_i_merge.fillna(0)

    # correct time due filtering ------------------------------------------
    #df_i['time'] = np.round(df_i['time'] - df_i['time'][0], 2)
    #df_vib['time'] = np.array(df_vib['time'] - df_vib['time'][0])
    DF  .append(df_i_merge) 

    # vibration acquisition -----------------------------------------------
    ac2 =  df_vib['AC-02'].values
    ac3 =  df_vib['AC-03'].values
    time = df_vib['time'].values
    AC2  = ac2  if k==0 else np.r_[AC2 , ac2 ]
    AC3  = ac3  if k==0 else np.r_[AC3 , ac3 ]
    TIME = time if k==0 else np.r_[TIME, time] 
    b.append(0) if k==0 else b.append(f[-1] + 1)
    f.append(b[k] + len(time) - 1)
    # mean of data --------------------------------------------------------
    df_mean_i = pd.DataFrame(df_i.mean()).T
    df_mean = df_mean_i if k==0 else pd.concat( 
      [df_mean, df_mean_i], axis=0, ignore_index=True)
    if k%100==0:
      print('%s/%s'%(k+1,k_max))
      time_elapsed(START)               # Time elapsed in the process
    k+=1
    del ac2, ac3, df_i, df_j, df, df_vib, exp, exp_vib, df_mean_i, time
    #if len(point.split('-'))>5:
    #fig, ax = plt.subplots(figsize=(20,20))
    #ax.plot(df_i['time'], df_i['ESP_rotation'], '--', color='gray')
    #ax.plot(df_i['time'], df_i['esp_rotation'], '--', color='k')
    #ax2 = ax.twinx()
    #ax2.plot(df_i['time'], df_i['choke_esp'], color='r')
    #fig.suptitle((curve, point), fontsize=20)
    #fig.savefig('figures/' + str(i) + '_' + point)
    #plt.close()
    #  plt.show()

print('last k is =', k)
time_elapsed(START)               # Time elapsed in the process
print('pahts to delete ', paths_del)
df_inf['vib_b'] = b
df_inf['vib_f'] = f 
print('constructed dataframe')
DF = pd.concat(DF, ignore_index=True, axis=0)
plt.plot(DF.time, DF.dp_ref)
plt.plot(DF.time, DF.P_9 - DF.P_1)
plt.plot(DF.time, DF.p9 - DF.p1)
plt.show()
plt.plot(DF.time, DF.Watercut)

# test matrix
# ==========================================================================
#exp = Sample(path);                 exp.set_data()
#df = pd.DataFrame(exp.data)


#%% =======================================================================
# save dataframe
# ==========================================================================
begin = 'ct_1ph_'
names = ['AC2.h5', 'AC3.h5', 'time.h5', 'DF.h5', 'df.h5', 'inf.h5']
names = [begin + i for i in names]

path = os.path.join('result', names[4]) 
dicttoh5(df_mean.to_dict('list'), path, create_dataset_args=d_s_a)  

path = os.path.join('result', names[5]) 
dicttoh5(df_inf.to_dict('list'), path)
time_elapsed(START)

path = os.path.join('result', names[3]) 
DF = pd.concat(DF, ignore_index=True, axis=0)
dicttoh5(DF.to_dict('list'), path, create_dataset_args=d_s_a)
del DF;    print('Imported DF');  time_elapsed(START)

path = os.path.join('result', names[0]) 
AC2 = pd.DataFrame({'AC-02': AC2})
dicttoh5(AC2.to_dict('list'), path, create_dataset_args=d_s_a)
del AC2;    print('Imported AC2');  time_elapsed(START)

path = os.path.join('result', names[1]) 
AC3 = pd.DataFrame({'AC-03': AC3})
dicttoh5(AC3.to_dict('list'), path, create_dataset_args=d_s_a)
del AC3;    print('Imported AC3');  time_elapsed(START)

path = os.path.join('result', names[2]) 
TIME = pd.DataFrame({'time': TIME})
dicttoh5(TIME.to_dict('list'), path, create_dataset_args=d_s_a)
del TIME;    print('Imported time');  time_elapsed(START)

time_elapsed(START)
