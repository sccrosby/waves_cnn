import numpy as np
import glob
import matplotlib.pyplot as plt
from data_utils.matlab_datenums import matlab_datenum_to_py_date as matlab_dtm
import pickle
import torch.nn as nn
import os

buoy_list = [46214] #, 46214, 46218]

fp = "datasets/forecast_data/"
time_vector = "forecast_time_zulu"

for buoy in buoy_list:
  npz_file = glob.glob(fp + str(buoy) + "_combined_data_*.npz")[-1]
  comb_data = np.load(npz_file)

  for_tensor = comb_data['forecast']
  obs_tensor = comb_data['observed']

  time = list(range(0, 190))

  print("buoy", buoy, "forecast shape: ", for_tensor[1,4,time[0]:time[-1]+1,100].shape)
  print("buoy", buoy, "observed shape: ", obs_tensor[1,4,time[0]:time[-1]+1].shape)
  print("x", len(time), "time[0]", time[0], "time[-1]", time[-1])

  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  ax1.scatter(time, for_tensor[1,4,time[0]:time[-1]+1,100], s=10, c='b', marker="s", label='forecast')
  ax1.scatter(time, obs_tensor[1,4,time[0]:time[-1]+1], s=10, c='r', marker="o", label='observed')
  plt.legend(loc='upper left');
  plt.savefig('plot.pdf')
