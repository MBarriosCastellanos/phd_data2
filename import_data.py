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
                  

# %% =======================================================================
# list curves
# ========================================================================== 
curves = sorted(glob.glob('data/ss*'))[:24]
curves_kind = []
k = 0
df_inf = pd.DataFrame(columns=['k', 'i', 'curve', 'point', 'curve_kind'])
for i, curve in enumerate(curves):
  print(('%s %s %s'%(i, curve, k)).center(70, '.'))
  curve_list = curve.split(sep)[-1].split('_')
  curve_kind = '_'.join(curve_list[:3])
  curves_kind.append(curve_kind)
  for path in sorted(glob.glob(curve + '/e*')):
    df_inf.loc[k, :] = [k, str(i), curve, 
      path.split(sep)[-1], curve_kind] 
    k+=1
curves_kind_unique = np.sort(np.unique(curves_kind))
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
for i, curve in enumerate(curves):
  for path in sorted(glob.glob(curve + '/e*')):
    printProgressBar(k + 1, k_max, prefix = '%s/%s'%(k+1,k_max), )
    # get process experiment -----------------------------------------------
    exp = Sample(path);                 exp.set_data()
    df = pd.DataFrame(exp.data)
    if np.shape(df)[0]!=3750 or np.shape(df)[1]!=33:
      paths_del.append(path)
      print(' incomplete path '.center(70, '█'))  
      continue

    # get vibration experiment ---------------------------------------------
    exp_vib = Sample_vib(path);     exp_vib.set_data()
    df_vib = pd.DataFrame(exp_vib.data)
    if np.shape(df_vib)[0]!=768000 or np.shape(df_vib)[1]!=8:
      paths_del.append(path)
      print(' incomplete path by vibration '.center(70, '█'))  
      continue

    # filetring of rotation sampling ---------------------------------------
    df['ESP_rotation'] = combined_filter(df['time'], df['ESP_rotation'], 
      10, 10/3, 20)
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
    df_i['curve_kind'] = np.where(curves_kind_unique==curve_kind)[0][0]
    df_vib['k'] = k

    # correct time due filtering ------------------------------------------
    df_i['time'] = np.round(df_i['time'] - df_i['time'][0], 2)
    df_vib['time'] = np.array(df_vib['time'] - df_vib['time'][0])
    DF  .append(df_i) 

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
    #del ac2, ac3, df_i, df_j, df, df_vib, exp, exp_vib, df_mean_i, time
print('last k is =', k)
time_elapsed(START)               # Time elapsed in the process
print('pahts to delete ', paths_del)
df_inf['vib_b'] = b
df_inf['vib_f'] = f 
print('constructed dataframe')


#%% =======================================================================
# save dataframe
# ==========================================================================
begin = 'ss_1ph'
names = ['AC2.h5', 'AC3.h5', 'time.h5', 'DF.h5', 'df.h5', 'inf.h5']
names = [begin + i for i in names]

path = os.path.join('result', names[4]) 
dicttoh5(df_mean.to_dict('list'), path, create_dataset_args=d_s_a)  

path = os.path.join('result', names[5]) 
dicttoh5(df_inf.to_dict('list'), path, create_dataset_args=d_s_a)
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


# %%
