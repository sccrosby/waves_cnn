import scipy.io
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from data_utils.load_data import load_both_as_dict
from datetime import datetime, timedelta

observed_file = 'data/CDIPObserved_46211_hourly.mat'
ww3_file  = 'data/WW3CFSRphase2_46211_rebanded_moments.mat'


def matlab_to_python_date(matlab_datenum):
    # http://sociograph.blogspot.com/2011/04/how-to-avoid-gotcha-when-converting.html
    python_date_time = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
    return python_date_time


def plot_timescales(buoy_time, glob_time):
    y_start = min(np.ndarray.min(buoy_time), np.ndarray.min(glob_time))
    plt.plot(glob_time, np.zeros_like(glob_time) + 0, 'x', marker=2)
    plt.plot(buoy_time, np.zeros_like(buoy_time) + 0, 'x', marker=3)
    plt.show()


def find_minmax_times(buoy_time, glob_time):
    buoy_min = np.ndarray.min(buoy_time)
    buoy_max = np.ndarray.max(buoy_time)

    print("Buoy Min:  %d" % buoy_min)
    print("Buoy Max:  %d" % buoy_max)

    glob_min = np.ndarray.min(glob_time)
    glob_max = np.ndarray.max(glob_time)

    print("Glob Min:  %d" % glob_min)
    print("Glob Max:  %d" % glob_max)

    print("Target Range: %d - %d" % (max(buoy_min, glob_min), min(buoy_max, glob_max)))


def plot_time_gaps_histogram(data, name="Default", num_bins=6):
    # this is very specific to CDIPObsered data at the moment
    time_vals = data['time']

    dims = time_vals.shape
    if dims[0] < dims[1]:
        time_vals = np.transpose(time_vals)

    time_vals = np.sort(time_vals)

    result = []
    result_ranges = []

    nans = 0

    # for i in range(time_vals.shape[0] - 1):
    i = 0
    while i < time_vals.shape[0] - 1:
        if np.isnan(data['b1'][0][i]):
            i += 1
            continue
        curr_val = matlab_to_python_date(time_vals[i][0])

        j = i + 1
        while j < time_vals.shape[0] - 1 and np.isnan(data['b1'][0][j]):
            j += 1

        next_val = matlab_to_python_date(time_vals[j][0])
        duration_in_s = (next_val - curr_val).total_seconds()
        duration_in_m = divmod(duration_in_s, 60)[0]
        if duration_in_m >= 61:
            result_ranges.append((curr_val, next_val))
            result.append(duration_in_m)
        i = j

    print(result_ranges)
    print(len(result))
    print(min(result))
    print(max(result))


    n, bins, patches = plt.hist(result, num_bins, facecolor='blue', alpha=0.5, rwidth=0.5)
    plt.title(name)
    plt.xlabel("minutes")
    plt.ylabel("count")
    plt.show()


if __name__ == '__main__':
    buoy_data, glob_data = load_both_as_dict(observed_file, ww3_file)

    print("Buoy: ")
    plot_time_gaps_histogram(buoy_data, name="46211_waves_TRAIN_2019-03-03")

    # print("Glob: ")
    # plot_time_gaps_histogram(glob_data, name="WW3")


    # buoy_time = buoy_data['time']
    # glob_time = glob_data['time']

    # PLOT
    # plot_timescales(buoy_data['time'], glob_data['time'])

    # Find Min/Max
    # Buoy Min:  728173
    # Buoy Max:  737211
    # Glob Min:  722816
    # Glob Max:  734138
    # Target Range: 728173 - 734138
    # find_minmax_times(buoy_time, glob_time)






# /home/mooneyj3/ml_waves/data/CDIPObserved_46211_hourly.mat
