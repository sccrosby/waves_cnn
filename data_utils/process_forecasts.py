import scipy.io
import time
import datetime
import numpy as np
from data_utils.matlab_datenums import matlab_datenum_to_py_date as matlab_dtm
import matplotlib.pyplot as plt
from datetime import datetime as dt

BUOY = 46214
REMOVENANS = True

forecast_file = "data/forecast_data/forecast_" + str(BUOY) + ".mat"
observed_file = "data/forecast_data/obs_" + str(BUOY) + ".mat"

forecast_data = scipy.io.loadmat(forecast_file)
observed_data = scipy.io.loadmat(observed_file)

class ForecastAttr:
  e  = forecast_data.get("e")
  a1 = forecast_data.get("a1")
  b1 = forecast_data.get("b1")
  a2 = forecast_data.get("a2")
  b2 = forecast_data.get("b2")

class ObservedAttr:
  e  = observed_data.get("e")
  a1 = observed_data.get("a1")
  b1 = observed_data.get("b1")
  a2 = observed_data.get("a2")
  b2 = observed_data.get("b2")

def removeNans():
  time_zulu = []
  obs_time = []
  #### CHECK FOR / REMOVE NANS FROM FORECAST ####
  print("NaN's in forecast data?",np.isnan(stacked_forecast[:, 4:31, :, :]).any())
  if np.isnan(stacked_forecast[:, 4:31, :, :]).any():
    print("removing time zulu indices with NaN's...")
    # get then number of time zulu indices where the data is NaN
    nan_idx = []
    for i in range(stacked_forecast.shape[-1]):
      if np.isnan(stacked_forecast[:, 4:31, :, i]).any():
        nan_idx.append(i)

    # initialize forecast tensor for data with no NaNs
    for_tensor = np.zeros((len(attributes), freq_len, for_time_len, time_zulu_len-len(nan_idx)))
    for_idx = 0

    # only save data with no NaNs
    for i in range(stacked_forecast.shape[-1]):
      if not np.isnan(stacked_forecast[:, 4:31, :, i]).any():
        for_tensor[:, :, :, for_idx] = stacked_forecast[:, :, :, i]
        time_zulu.append(forecast_data.get("time_zulu")[0,i])
        for_idx = for_idx + 1

    print("removed", len(nan_idx), "indices")
    print("\nNaN's in forecast data?",np.isnan(for_tensor[:, 4:31, :, :]).any())
    print("shape of full forecast data:", stacked_forecast.shape)
    print("shape after removing NaN's: ", for_tensor.shape)
  else:
    for_tensor = stacked_forecast
    time_zulu = forecast_data.get("time_zulu")[0, :]
    print("shape of forecast data:", for_tensor.shape)
  print("forecast_time_zulu:", len(time_zulu))

  #### CHECK FOR / REMOVE NAN'S FROM OBSERVED ####
  print("\nNaN's in observed data?",np.isnan(stacked_observed[:, 4:31, :]).any())
  if np.isnan(stacked_observed[:, 4:31, :]).any():
    print("removing time indices with NaN's...")
    # get then number of time zulu indices where the data is NaN
    nan_idx = []
    for i in range(stacked_observed.shape[-1]):
      if np.isnan(stacked_observed[:, 4:31, i]).any():
        nan_idx.append(i)

    # initialize forecast tensor for data with no NaNs
    obs_tensor = np.zeros((len(attributes), freq_len, obs_time_len-len(nan_idx)))
    obs_idx = 0

    # only save data with no NaNs
    for i in range(stacked_observed.shape[-1]):
      if not np.isnan(stacked_observed[:, 4:31, i]).any():
        obs_tensor[:, :, obs_idx] = stacked_observed[:, :, i]
        obs_time.append(observed_data.get("time")[0,i])
        obs_idx = obs_idx + 1

    print("removed", len(nan_idx), "indices")
    print("\nNaN's in observed data?",np.isnan(obs_tensor[:, 4:31, :]).any())
    print("shape of full observed data:", stacked_observed.shape)
    print("shape after removing NaN's: ", obs_tensor.shape)
  else:
    obs_tensor = stacked_observed
    obs_time = observed_data.get("time")[0, :]
    print("shape of observed data:", obs_tensor.shape)
  print("observed_time:", len(obs_time))

  return for_tensor, obs_tensor, time_zulu, obs_time


for_time_len = forecast_data.get("e").shape[1]
obs_time_len = observed_data.get("time").shape[-1]
freq_len = forecast_data.get("fr").shape[0]
time_zulu_len = forecast_data.get("time_zulu").shape[-1]
attributes = ["a1", "b1", "a2", "b2", "e"]

print("buoy", BUOY)
print("forecast")
print("\te   ",forecast_data.get("e").shape)
print("\ta1  ",forecast_data.get("a1").shape)
print("\tb1  ",forecast_data.get("b1").shape)
print("\ta2  ",forecast_data.get("a2").shape)
print("\tb2  ",forecast_data.get("b2").shape)
print("\tfr  ",forecast_data.get("fr").shape)
print("\tbw  ",forecast_data.get("bw").shape)
print("\ttime",forecast_data.get("time_zulu").shape)

print("observed")
print("\te   ",observed_data.get("e").shape)
print("\ta1  ",observed_data.get("a1").shape)
print("\tb1  ",observed_data.get("b1").shape)
print("\ta2  ",observed_data.get("a2").shape)
print("\tb2  ",observed_data.get("b2").shape)
print("\tfr  ",observed_data.get("fr").shape)
print("\tbw  ",observed_data.get("bw").shape)
print("\ttime",observed_data.get("time").shape)
# stack forecast data into (5, 64, forecast_time, time_zulu) shape
# and stack observed data into (5, 64, time) shape
stacked_forecast = np.empty((len(attributes), freq_len, for_time_len, time_zulu_len))
stacked_observed = np.empty((len(attributes), freq_len, obs_time_len))

# obs: time x 5 x 64
for a,attr in enumerate(attributes):
  stacked_forecast[a] = getattr(ForecastAttr, attr)
  stacked_observed[a] = getattr(ObservedAttr, attr)


if REMOVENANS:
  for_tensor, obs_tensor, time_zulu, obs_time = removeNans()
else:
  for_tensor = stacked_forecast  
  obs_tensor = stacked_observed
  time_zulu = forecast_data.get("time_zulu")[0,:]
  obs_time = observed_data.get("time")[:,0]


# Forecast data is in 3-hour intervals. Interpolate to hourly
new_size = for_tensor.shape[]

#print(obs_time[0:5])
#print(obs_tensor[0, 4:31, :].shape)
#for i in range(obs_tensor.shape[1]):
#  if (i >= 4 and i <= 31):
#    plt.plot(obs_time, obs_tensor[0, i, :])
#    plt.savefig('plots/datarange' + str(i) + '.png')


#    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)
#
#for_time = [str(matlab_dtm(time_zulu[0])).split(".")[0]]
#for i in range(1,for_tensor.shape[2]):
#  for_time.append(for_time[i-1] + 0.0416666666678)
#
#for t in for_time:
#  print(str(matlab_dtm(t)))


print("saving data...")
np.savez('datasets/forecast_data/' + str(BUOY) + '_combined_data_' + str(datetime.date.today()) + '.npz',
         forecast_time_zulu=time_zulu,
         observed_time=obs_time,
         forecast=for_tensor,
         observed=obs_tensor)
print("done saving data")

# inputs:
# history_len forecast_len
# +-----------+-----------+
# | observed  | forecast 0|
# +-----------+-----------+
# |  forecast -h          |
# +-----------------------+
#
# residuals = forecast 0
#
# targets:
# +-----------+-----------+
# |           | observed  |
# +-----------+-----------+


# obs_oldest = np.min(observed_data.get("time"))
# obs_newest = np.max(observed_data.get("time"))
# for_oldest = np.min(forecast_data.get("time_zulu"))
# for_newest = np.max(forecast_data.get("time_zulu"))

# oldest_date = max(obs_oldest, for_oldest)
# newest_date = min(obs_newest, for_newest)

