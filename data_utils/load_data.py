import scipy.io
import time
import numpy as np

log_times = True


def load_both_as_dict(buoy_filename, global_filename):
    if log_times:
        start = time.time()

    buoy_data = scipy.io.loadmat(buoy_filename)
    glob_data = scipy.io.loadmat(global_filename)

    if log_times:
        runtime = time.time() - start
        print("Time to load files: %d s" % runtime)

    return buoy_data, glob_data


def load_data_as_dict(filename):
    data = scipy.io.loadmat(filename)
    return data
