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
from ConvNN.cnn import ConvNN
from data_utils.data_processing import save_single_set_to_npz as npz_save
from data_utils.data_processing import convert_to_third_order_tensor as convert_tensor

buoy = str(46015)

npz_file = "datasets/new_buoys/" + buoy + "_combined_data_2021-05-27.npz"

# Load the data from npz
comb_data = np.load(npz_file,allow_pickle=True)

# Time vector representing all matlab serialized date/times
time_vector = comb_data['time_vector']

print("first value: ", str(matlab_dtm(time_vector[0])))
print("last value:  ", str(matlab_dtm(time_vector[-1])))
print("total length:", len(time_vector))

year_1_end = 8766
last_year = -8766

print("\nfirst year:",  str(matlab_dtm(time_vector[0])), " -- to -- ", str(matlab_dtm(time_vector[year_1_end])))
print("middle:    ",  str(matlab_dtm(time_vector[year_1_end])), " -- to -- ", str(matlab_dtm(time_vector[last_year])))
print("last year: ",  str(matlab_dtm(time_vector[last_year])), " -- to -- ", str(matlab_dtm(time_vector[-1])))

# Observed Tensor - data processed from buoy
obs_tensor = comb_data['obs_tensor']

# WW3 Tensor - data processed from hindcast data
ww3_tensor = comb_data['ww3_tensor']

save = True

if save:
  print("saving data...")

  # save train
  train_npz = 'datasets/new_buoys/' + buoy + '_waves_TRAIN_' + str(datetime.date.today()) + '.npz'
  npz_save(train_npz, time_vector[0:year_1_end], obs_tensor[0:year_1_end], ww3_tensor[0:year_1_end])

  # save test
  test_npz = 'datasets/new_buoys/' + buoy + '_waves_TEST_' + str(datetime.date.today()) + '.npz'
  npz_save(test_npz, time_vector[year_1_end:last_year], obs_tensor[year_1_end:last_year], ww3_tensor[year_1_end:last_year])

  # save dev
  dev_npz = 'datasets/new_buoys/' + buoy + '_waves_DEV_' + str(datetime.date.today()) + '.npz'
  npz_save(dev_npz, time_vector[last_year:-1], obs_tensor[last_year:-1], ww3_tensor[last_year:-1])
  print("...done saving data")
