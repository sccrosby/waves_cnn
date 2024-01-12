import numpy as np
from data_utils.matlab_datenums import matlab_datenum_to_py_date as matlab_dtm
from data_utils.constants import FREQUENCY_RANGE, BANDWIDTH_RANGE, E, A1, B1, A2, B2
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import os
import scipy.io
import datetime
import glob
from ConvNN.cnn import ConvNN
from data_utils.data_processing import save_single_set_to_npz as npz_save
from data_utils.data_processing import convert_to_third_order_tensor as convert_tensor

buoy = str(46015)

npz_file = glob.glob("datasets/forecast_data/" + buoy + "_combined_data_*.npz")[-1]

# Load the data from npz
comb_data = np.load(npz_file,allow_pickle=True)

# Time vector representing all matlab serialized date/times
forecast_time_zulu = comb_data['forecast_time_zulu']
observed_time = comb_data['observed_time']

# Forecast and observed tensors
for_tensor = comb_data['forecast']
obs_tensor = comb_data['observed']

for i,date in enumerate(observed_time):
  if (i-1 > 0 and (date-observed_time[i-1] > 0.05)):
    print(i)
    print("\t",str(matlab_dtm(observed_time[i-1])))
    print("\t",str(matlab_dtm(date)))



for i,date in enumerate(forecast_time_zulu):
  found_obs_date = np.where(observed_time <= date+0.01)[0]
  if (len(found_obs_date) > 0 and (found_obs_date[-1]-date < 0.05)):
    for_start_date = i
    obs_start_date = found_obs_date[-1]    
    break

#if (buoy == "46015"):
#  for_start_date = 221
#elif (buoy == "46027"):
#  for_start_date = 3
#elif (buoy == "46029"):
#  for_start_date = 2
#elif (buoy == "46089"):
#  for_start_date = 1
#else:
#  for_start_date = 0

#for_year_1_end = for_len//3
#for_year_2_end = -for_len//3
#for_end_date   = -1

for_len = forecast_time_zulu.shape[0] - for_start_date
obs_len = observed_time.shape[0]

for i,date in enumerate(forecast_time_zulu[for_len//3+for_start_date:for_len]):
  obs_dates = np.where(observed_time <= date+0.01)[0]
  if len(obs_dates) > 0 and (date - obs_dates[-1] < 0.05):
    for_year_1_end = i
    obs_year_1_end = obs_dates[-1]
    break
    
for_year_2_end = -for_len//3
for_end_date   = -1
#print(str(matlab_dtm(forecast_time_zulu[for_start_date])), ", ", str(matlab_dtm(observed_time[0])))
obs_start_date = np.where(observed_time <= forecast_time_zulu[for_start_date]+0.01)[0][-1]

obs_year_1_end = np.where(observed_time <= forecast_time_zulu[for_year_1_end]+0.01)[0][-1]
obs_year_2_end = np.where(observed_time <= forecast_time_zulu[for_year_2_end]+0.01)[0][-1]
obs_end_date   = np.where(observed_time <= forecast_time_zulu[for_end_date]+0.01)[0][-1]


print("forecast_time_zulu  ", str(matlab_dtm(forecast_time_zulu[for_year_1_end])))
print("observed_time before", str(matlab_dtm(observed_time[obs_year_1_end])))
print("observed_time after?", str(matlab_dtm(observed_time[obs_year_1_end+1])))



print("\nfirst forecast year:", str(matlab_dtm(forecast_time_zulu[for_start_date])),
                  " -- to -- ", str(matlab_dtm(forecast_time_zulu[for_year_1_end])))
print("first observed year:", str(matlab_dtm(observed_time[obs_start_date])),
                  " -- to -- ", str(matlab_dtm(observed_time[obs_year_1_end])))

print("\nmiddle forecast:    ",   str(matlab_dtm(forecast_time_zulu[for_year_1_end])),
                  " -- to -- ", str(matlab_dtm(forecast_time_zulu[for_year_2_end])))
print("middle observed:    ",   str(matlab_dtm(observed_time[obs_year_1_end])),
                  " -- to -- ", str(matlab_dtm(observed_time[obs_year_2_end])))

print("\nlast forecast year: ",   str(matlab_dtm(forecast_time_zulu[for_year_2_end])),
                  " -- to -- ", str(matlab_dtm(forecast_time_zulu[for_end_date])))
print("last observed year: ",   str(matlab_dtm(observed_time[obs_year_2_end])),
                  " -- to -- ", str(matlab_dtm(observed_time[obs_end_date])))

save = True
print("time zulu", len(forecast_time_zulu))
print("FORECAST", for_tensor.shape)
print("OBSERVED", obs_tensor.shape)

if save:
  print("saving data for buoy", buoy, "...")

  # save train
  train_npz = 'datasets/forecast_data/' + buoy + '_waves_TRAIN_' + str(datetime.date.today()) + '.npz'
  np.savez(train_npz, forecast_time_zulu=forecast_time_zulu[for_start_date:for_year_1_end],
                      observed_time=observed_time[obs_start_date:obs_year_1_end],
                      forecast=for_tensor[:, :, :, for_start_date:for_year_1_end],
                      observed=obs_tensor[:, :, obs_start_date:obs_year_1_end])

  # save test
  test_npz = 'datasets/forecast_data/' + buoy + '_waves_TEST_' + str(datetime.date.today()) + '.npz'
  np.savez(test_npz, forecast_time_zulu=forecast_time_zulu[for_year_1_end:for_year_2_end],
                     observed_time=observed_time[obs_year_1_end:obs_year_2_end],
                     forecast=for_tensor[:, :, :, for_year_1_end:for_year_2_end],
                     observed=obs_tensor[:, :, obs_year_1_end:obs_year_2_end])

  # save dev
  dev_npz = 'datasets/forecast_data/' + buoy + '_waves_DEV_' + str(datetime.date.today()) + '.npz'
  np.savez(dev_npz, forecast_time_zulu=forecast_time_zulu[for_year_2_end:for_end_date],
                    observed_time=observed_time[obs_year_2_end:obs_end_date],
                    forecast=for_tensor[:, :, :, for_year_2_end:for_end_date],
                    observed=obs_tensor[:, :, obs_year_2_end:obs_end_date])
  print("...done saving data")
