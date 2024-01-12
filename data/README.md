# ml_waves Data

All data is available in `/home/hutch_research/data/waves`

## Source

TODO: notes about data
```
Description of files
Obs and Predictions at NOAA NDBC buoy location 46211: https://www.ndbc.noaa.gov/station_page.php?station=46211

Link to original data: https://wwu2-my.sharepoint.com/:f:/r/personal/crosbys4_wwu_edu/Documents/WaveForecasting_CS?csf=1&e=6loEDO

WW3CFSRphase2_46211_rebanded.mat
Overview: Wave spectra predictions from WW3 CFSR Phase 2 reanalysis at NOAA buoy site 46211, Gray's Harbor, offshore of WA
Variables
- time (90584x1): Matlab vector datenum format, dt = 3-hour, 1979-2009.
- sp (64x36x90584): Wave energy density spectra, E(freq,theta,time), units: m^2/(Hz-deg)
- dir (36x1): Direction vector for theta, units: deg
- fr (64x1): Frequency vector, units: Hz
- bw  (64x1): Bandwidth, units Hz
- e: Wave energy freq spectra, integrated from sp. units: m^2/Hz
- hs: Significant wave height, units: m
- dtheta (1x1): 10
- lat, lon (1x1): fixed values

036p1_historic.nc
- time (1x216936): Matlab vector datenum format, dt = 1-hour, 1993-Present. However data gaps exist, and e,a1,b1 etc are filled with NaN
- e: Wave energy freq spectra, units: m^2/Hz
- a1 (64x216936) = first order directional moment, a1(f) = int[E(f,theta)*cos(theta) dtheta]
- b1 (64x216936) = second order directional moment, b1(f) = int[E(f,theta)*sin(theta) dtheta]
- a2 (64x216936) = first order directional moment, a1(f) = int[E(f,theta)*cos(2*theta) dtheta]
- b2 (64x216936) = second order directional moment, b1(f) = int[E(f,theta)*sin(2*theta) dtheta]
- fr (64x1): frequency, units: Hz
- bw (64x1): bandwidth, units: Hz

```

## Processing with matlab sample code

## Pre-processing libraries

## About the Processed data

All data is stored in a compressed numpy zip called `comp_combined_data_2019-02-15.npz`, created by `data_utils.data_processing`

This can be loaded in with numpy.load.

The data has the following attributes:
```
import numpy as np
file = 'data/comp_combined_data_2019-02-15.npz'
data = np.load(file)

# Time vector representing all matlab serialized date/times
time_vector = data['time_vector']

# Observed Tensor - data processed from buoy
obs_tensor = data['obs_tensor']

# WW3 Tensor - data processed from hindcast data
ww3_tensor = data['ww3_tensor']
```

#### Shape of data

__time_vector__
* shape: (N,)
* Simple numpy array of serial dates

__obs_tensor and ww3_tensor__
* shape: (N, 5)
* The 5 denotes the depth of the stack for `a1, a2, b1, b2, e` in the original dataset
* Indexing at N\[x\] will get into the dimensionality at a specific time
  * example:  `obs_tensor[0][0]` is the `a1` and `obs_tensor[0][1]` is the `a2` matrix at the first serial date in the observed tensor
  * Hence, `obs_tensor[0].shape` is (5, ), and `obs_tensor[0][0].shape` is (64,)

#### TRAIN, DEV, TEST Outputs

Slicing data from data/comp_combined_data_2019-02-15.npz, where we have \[start_index, end_index\]
* Train \[0, 76543\]
* Dev \[76543, 93491\]
* Test \[93491, 109305\]

Extending from the code above, the data was saved as follows
```
from data_utils.data_processing import save_single_set_to_npz as npz_save
import datetime

# save train
train_npz = 'waves_TRAIN_' + str(datetime.date.today()) + '.npz'
npz_save(train_npz, time_vector[0:76543], obs_tensor[0:76543], ww3_tensor[0:76543])

# save dev
dev_npz = 'waves_DEV_' + str(datetime.date.today()) + '.npz'
npz_save(dev_npz, time_vector[76543:93491], obs_tensor[76543:93491], ww3_tensor[76543:93491])

# save test
test_npz = 'waves_TEST_' + str(datetime.date.today()) + '.npz'
npz_save(test_npz, time_vector[93491:109306], obs_tensor[93491:109306], ww3_tensor[76543:109306])
```

#### Converting Serial Times

This is a handy routine.
I use it to verify dates and times are properly aligned where I expect them

```
from data_utils.matlab_datenums import matlab_datenum_to_py_date as mdtm

print(mdtm(train_time_vector[-1]))
# ouputs: 2003-04-07 15:59:59.999997
```

## Data Stats
