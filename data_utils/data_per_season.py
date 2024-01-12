import numpy as np
import glob
import seaborn as sb
import matplotlib.pyplot as plt
from data_utils.matlab_datenums import matlab_datenum_to_py_date as matlab_dtm
import pickle
import torch.nn as nn
import os

class QuarterlyTotals:
  def __init__(self):
    self.q1_total = Total(0)
    self.q2_total = Total(0)
    self.q3_total = Total(0)
    self.q4_total = Total(0)
    self.totals_by_quarter = { "01": self.q1_total,
                               "02": self.q1_total,
                               "03": self.q1_total,
                               "04": self.q2_total,
                               "05": self.q2_total,
                               "06": self.q2_total,
                               "07": self.q3_total,
                               "08": self.q3_total,
                               "09": self.q3_total,
                               "10": self.q4_total,
                               "11": self.q4_total,
                               "12": self.q4_total }

class Total:
  def __init__(self, value = None):
    self.value = value
 
def get_data(isForecast, buoy_list):
  train = np.zeros((len(buoy_list),4))
  dev = np.zeros((len(buoy_list),4))
  test = np.zeros((len(buoy_list),4))
  hrs = 24

  for i,buoy in enumerate(buoy_list):
    fp = "datasets/"
    time_vector = "time_vector"
    if isForecast:
      fp = fp+"forecast_data/"
      time_vector = "forecast_time_zulu"
      hrs = 4
    elif (buoy==46015 or buoy==46027 or buoy==46029 or buoy==46089):
      fp = fp+"new_buoys/"

    npz_train = glob.glob(fp + str(buoy) + "_waves_TRAIN_*.npz")[-1]
    npz_dev = glob.glob(fp + str(buoy) + "_waves_DEV_*.npz")[-1]
    npz_test = glob.glob(fp + str(buoy) + "_waves_TEST_*.npz")[-1]
    print(fp + str(buoy) + "_waves_TRAIN_*.npz")

    # Load the data from npz
    train_data = np.load(npz_train)
    dev_data = np.load(npz_dev)
    test_data = np.load(npz_test)

    # Time vector representing all matlab serialized date/times
    train_time = train_data[time_vector]
    dev_time = dev_data[time_vector]
    test_time = test_data[time_vector]

    print("dd\n\ntime\n")
    for t in train_time:
      print(str(matlab_dtm(t)))

    print("buoy", buoy)
    print("\ttrain len", len(train_time))
    print("\tdev len", len(dev_time))
    print("\ttest len", len(test_time))
  
    # Data dictionary
    train_quarterly_totals = QuarterlyTotals()
    dev_quarterly_totals = QuarterlyTotals()
    test_quarterly_totals = QuarterlyTotals()

    for time in train_time:
      time_str = str(matlab_dtm(time))[5:7]
      train_quarterly_totals.totals_by_quarter[time_str].value += 1 

    for time in dev_time:
      time_str = str(matlab_dtm(time))[5:7]
      dev_quarterly_totals.totals_by_quarter[time_str].value += 1 

    for time in test_time:
      time_str = str(matlab_dtm(time))[5:7]
      test_quarterly_totals.totals_by_quarter[time_str].value += 1 

    train[i,0]  = train_quarterly_totals.q1_total.value
    train[i,1]  = train_quarterly_totals.q2_total.value
    train[i,2]  = train_quarterly_totals.q3_total.value
    train[i,3]  = train_quarterly_totals.q4_total.value

    dev[i,0]  = dev_quarterly_totals.q1_total.value
    dev[i,1]  = dev_quarterly_totals.q2_total.value
    dev[i,2] = dev_quarterly_totals.q3_total.value
    dev[i,3] = dev_quarterly_totals.q4_total.value

    test[i,0]  = test_quarterly_totals.q1_total.value
    test[i,1]  = test_quarterly_totals.q2_total.value
    test[i,2]  = test_quarterly_totals.q3_total.value
    test[i,3]  = test_quarterly_totals.q4_total.value

  return np.rint(train/hrs).astype(int), np.rint(dev/hrs).astype(int), np.rint(test/hrs).astype(int)

def make_plot(train, dev, test, buoys, plot_title, figsize, center, vmin, vmax):
  seasons = ["Winter", "Spring", "Summer", "Fall"]

  print("making plot... ")
  f,(ax1,ax2,ax3,axcb) = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,0.08]},
                                      figsize=figsize)
  ax1.get_shared_y_axes().join(ax2,ax3)
  ax2.set_title(plot_title)

  g1 = sb.heatmap(train,
                  cmap='YlGnBu',
                  xticklabels=seasons,
                  yticklabels=buoys,
                  annot=True,
#                  annot_kws={"size": 8},
                  fmt="d",
                  center=center,
                  vmin = vmin,
                  vmax = vmax,
                  square=True,
                  cbar=False,
                  ax=ax1)
  g1.set_xlabel("TRAIN")

  g2 = sb.heatmap(dev,
                  cmap='YlGnBu',
                  xticklabels=seasons,
                  yticklabels=False,
                  annot=True,
#                  annot_kws={"size": 8},
                  fmt="d",
                  center=center,
                  vmin = vmin,
                  vmax = vmax,
                  square=True,
                  cbar=False,
                  ax=ax2)
  g2.set_xlabel("DEV")

  g3 = sb.heatmap(test,
                  cmap='YlGnBu',
                  xticklabels=seasons,
                  yticklabels=False,
                  annot=True,
#                  annot_kws={"size": 8},
                  fmt="d",
                  center=center,
                  vmin = vmin,
                  vmax = vmax,
                  square=True,
                  ax=ax3,
                  cbar_ax=axcb,
                  cbar_kws={'label': 'days of data'})
  g3.set_xlabel("TEST")

  plt.savefig('plt.pdf')
  print("done making plot")




data_type = "forecast"
#data_type = "hindcast"
#data_type = "new_hindcast"
#data_type = "old_hindcast"

train = dev = test = np.zeros((0,0))
buoys = []
plot_title = ""
figsize = (0,0)
center = ""
vmin = vmax = 0

print("getting data per season... ")
if (data_type == "forecast"):
  train, dev, test = get_data(True, [46211, 46214, 46218])
  buoys = ["Grays Harbor", "Point Reyes", "Harvest"]
  plot_title = "Forecast Dataset"
  figsize = (12,4)
  vmin = 0
  vmax = min(int(max(np.max(train), np.max(dev), np.max(test))), 365)
  center = vmax//2

elif (data_type == "hindcast"):
  train,dev,test = get_data(False, [46211, 46029, 46089, 46015, 46027, 46214, 46218])
  buoys = ["Grays Harbor", "Columbia River", "Tillamook", "Port Orford",
           "Crescent City", "Point Reyes", "Harvest"]
  plot_title = "Hindcast Dataset"
  figsize = (12,6)
  vmin = 0
  vmax = min(int(max(np.max(train), np.max(dev), np.max(test))), 365)
  center = vmax//2

elif (data_type == "new_hindcast"):
  train, dev, test = get_data(False, [46029, 46089, 46015, 46027])
  buoys = ["Columbia River", "Tillamook", "Port Orford", "Crescent City"]
  plot_title = "Hindcast Dataset"
  figsize = (15,5)
  vmin = 0
  vmax = int(max(np.max(train), np.max(dev), np.max(test)))
  center = vmax//2

elif (data_type == "old_hindcast"):
  train, dev, test = get_data(False, [46211, 46214, 46218])
  buoys = ["Grays Harbor", "Point Reyes", "Harvest"]
  plot_title = "Hindcast Dataset"
  figsize = (12,4)
  vmin = 0
  vmax = min(int(max(np.max(train), np.max(dev), np.max(test))), 365)
  center = vmax//2

print(train)
print(dev)
print(test)
print("done getting data")
#make_plot(train, dev, test, buoys, plot_title, figsize, center, vmin, vmax)
