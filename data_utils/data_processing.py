import data_utils.load_data as load_data
import numpy as np
import time
import datetime
from data_utils.constants import A1, A2, B1, B2, E
from data_utils.matlab_datenums import matlab_datenum_to_py_date as matlab_dtm
import scipy.io

# from data_utils.data_processing import ObservedDatasetFromFile, WW3DatasetFromFile
# buoy = ObservedDatasetFromFile('data/CDIPObserved_46211_hourly.mat')
# glob = WW3DatasetFromFile('data/WW3CFSRphase2_46211_rebanded.mat')

# python console
# execfile('data_utils/data_processing.py')


class DatasetFromFile:
    """DatasetFromFile class
    Generic implementation to load data
    """
    def __init__(self, data_file, source_type=None):
        """
        Takes in a data file that is expected to be ".mat" format
        It will convert the raw data (loaded as dict to class objects.

        :param data_file:
        :param source_type:
        """
        # Load raw_data from file name
        self.raw_data = load_data.load_data_as_dict(data_file)

        # convert raw data to class objects
        self.__dict__.update(self.raw_data)

        # extract min/max dates
        if hasattr(self, 'time'):
            self.min_date = np.min(self.time)
            self.max_date = np.max(self.time)
            self.n = np.max(self.time.shape)
        else:
            raise Exception('DatasetFromFile object creation error: data has no `time` attribute')

    def check_class_attributes(self, attr_list):
        """
        Check for class attributes within the attribute list argument.

        :param attr_list:
        :return:
        """
        if not all(hasattr(self, attr) for attr in attr_list):
            missing = []
            for item in attr_list:
                missing.append(item) if item not in self.__dir__() else None
            raise Exception('DatasetFromFile object creation error: missing one or more of the following class attributes:' +
                            '\n\t Expected:\t' + str(attr_list) +
                            '\n\t Missing:\t' + str(missing))

    def combine_data(self, attributes):
        """
        This method combines data along the time axis.

        Time should be a given class attribute from data that is loaded
        Typically, it should be receiving this list: ['a1', 'a2', 'b1', 'b2', 'e']
        Unless we enhance this further to extend to other wave data.

        :param attributes: list of class attributes to stack
        :return:
            List of data with data points:  [time, [attr[0]], ..., [attr[n]]]
            List of time gaps with points:  [[start date, end date], [start index, end index]]
                Note: the time gaps are represented by a serial date.  The start date will have
                    Nan data, the end date will have actual data
        """
        # check the class attributes are available before proceeding
        if not hasattr(self, 'time'):
            raise AttributeError("combine_data missing attribute: time")
        for attr in attributes:
            if not hasattr(self, attr):
                raise AttributeError("combine_data missing attribute: " + attr)

        # get number of entries
        # n = getattr(self, 'time').shape[0]

        # carcass of resulting array
        stacked_data = np.empty((self.n, len(attributes) + 1), dtype=object)

        # setup time gaps
        time_gaps = []

        # inject time along the 0 axis of the result array
        stacked_data[:, 0] = self.time.reshape((self.n,))

        in_gap = False
        # iterate through all data entries (n)
        for i in range(self.n):
            # iterate through each attribute to stack the data
            for j in range(1, len(attributes) + 1):
                curr_attr = getattr(self, attributes[j-1])[i, :]
                # Check if we have empty data in the frequency range 0.04 - 0.25
                if np.isnan(curr_attr[4:31]).any():
                    # if we are entering a gap for the first time, then we need to get the start details
                    if not in_gap:
                        gap_start_index = i
                        gap_start_value = stacked_data[i, 0]
                        in_gap = True
                    # set all values to NaN for the current setup
                    for k in range(1, len(attributes) + 1):
                        stacked_data[i, k] = np.nan
                    continue

                # otherwise, close the gap and append it to gap list
                if in_gap:
                    print(curr_attr)
                    dates = [gap_start_value, stacked_data[i, 0]]
                    indices = [gap_start_index, i]
                    time_gaps.append([dates, indices])
                    gap_start_index = float("inf")
                    gap_start_value = float("inf")
                    in_gap = False

                # add the result
                stacked_data[i, j] = curr_attr

        return stacked_data, time_gaps

    def get_min_max_dates(self):
        min_date = np.min(getattr(self, 'time'))
        max_date = np.max(getattr(self, 'time'))
        return min_date, max_date

class ObservedDatasetFromFile(DatasetFromFile):
    """
    Abstract class for processing Matlab files containing wave data.
    Buoy DatasetFromFile (Observed
    DatasetFromFile contains:
        a1      (64, N)
        a2      (64, N)
        b1      (64, N)
        b2      (64, N)
        bw      (64, 1)
        e       (64, N)
        fr      (64, 1)
        time    (1, N)
    """
    def __init__(self, data_file):
        # load file
        super(ObservedDatasetFromFile, self).__init__(data_file)

        # check expected attributes
        expected_attributes = ['a1', 'a2', 'b1', 'b2', 'bw', 'e', 'fr', 'time']
        self.check_class_attributes(expected_attributes)

        # transpose all data attributes
        for attr in expected_attributes:
            setattr(self, attr, getattr(self, attr).transpose())

        # generate stacked data
        self.stacked_data, self.time_gaps = self.combine_data(['a1', 'a2', 'b1', 'b2', 'e'])


class WW3DatasetFromFile(DatasetFromFile):
    """Global DatasetFromFile (WW3)

    DatasetFromFile contains:
        bw      (64, 1)
        dir     (36, 1)
        dtheta  int/float
        fr      (64, 1)
        lat     float
        lon     float
        sp      (64, 36, N)
        time    (N, 1)
    """
    def __init__(self, data_file):
        # load file
        super(WW3DatasetFromFile, self).__init__(data_file)

        # check for expected attributes from loading data
        # expected_attributes = ['bw', 'dir', 'dtheta', 'fr', 'lat', 'lon', 'sp', 'time', 'a1', 'a2', 'b1', 'b2', 'e']
        expected_attributes = ['bw', 'fr', 'time', 'a1', 'a2', 'b1', 'b2', 'e']

        self.check_class_attributes(expected_attributes)

        # transpose attributes
        for attr in expected_attributes:
            if attr in ['time', 'dtheta', 'lat', 'lon']:
                continue
            setattr(self, attr, getattr(self, attr).transpose())

        # generate stacked data
        self.stacked_data, self.time_gaps = self.combine_data(['a1', 'a2', 'b1', 'b2', 'e'])

        # generate hourly data
        self.stacked_data_hourly = self.stacked_data_to_hourly()

    def stacked_data_to_hourly(self):
        """
        WW3 data is in a 3-hour interval.  This method interpolates to
        an hourly interval using averaging methods.
        """
        # set the new size of the output tensor
        new_size = (self.n - 1) * 3 + 1

        # create an empty result object to put the data into
        result = np.empty((new_size, 6), dtype=object)

        # get a 1/3 matrix of stacked_data
        stacked_third = 1./3. * self.stacked_data

        # iterate through array
        i = 0
        while i < self.n - 1:
            # extract current and next
            current = stacked_third[i]
            next = stacked_third[i + 1]

            # t1 = 2/3 * current + 1/3 * next
            # t2 = 1/3 * current + 2/3 * next
            t1 = 2 * current + next
            t2 = current + 2 * next
            # append current, t1, and t2 to the result matrix
            result[i * 3] = self.stacked_data[i]
            result[i * 3 + 1] = t1
            result[i * 3 + 2] = t2

            # increment
            i += 1

        # add last item to list
        result[-1] = self.stacked_data[-1]

        return result


def generate_combined_datasets(observed_files, ww3_files):
    """
    global tensor (Nx5x64) [only include data for valid hours]
    buoy tensor (Nx5x64) [only include data for valid hours]
    time vector (N) [only include times for valid hours]

    :return:
    """
    # Convert files to class objects
    obs_datasets = []
    ww3_datasets = []
    if observed_files is list:
        for file in observed_files:
            obs_datasets.append(ObservedDatasetFromFile(file))
    else:
        obs_datasets.append(ObservedDatasetFromFile(observed_files))
    if ww3_files is list:
        for file in ww3_files:
            ww3_datasets.append(WW3DatasetFromFile(file))
    else:
        ww3_datasets.append(WW3DatasetFromFile(ww3_files))

    # Extract min and max dates
    obs_oldest = obs_newest = np.nan
    ww3_oldest = ww3_newest = np.nan
    for obs in obs_datasets:
        old, new = obs.get_min_max_dates()
        obs_oldest = max(old, obs_oldest)
        obs_newest = min(new, obs_newest)
    for ww3 in ww3_datasets:
        old, new = ww3.get_min_max_dates()
        ww3_oldest = max(old, ww3_oldest)
        ww3_newest = min(new, ww3_newest)

    # combined date range
    print("obs_oldest", obs_oldest)
    print("ww3_oldest", ww3_oldest)
    oldest_date = max(obs_oldest, ww3_oldest)
    newest_date = min(obs_newest, ww3_newest)

    # Collect all the time gaps across all objects
    all_time_gaps = []
    for item in (obs_datasets + ww3_datasets):
        all_time_gaps += [a[0] for a in item.time_gaps]

    # sort all_time_gaps so we can coalesce overlaps
    all_time_gaps.sort()
    merged_time_gaps = merge_time_gaps(all_time_gaps)

    # remove gaps outside of range of all datasets
    abs_time_gaps = []
    for i in range(len(merged_time_gaps)):
        start_gap = merged_time_gaps[i][0]
        end_gap = merged_time_gaps[i][1]
        if start_gap > newest_date:
            continue
        elif end_gap < oldest_date:
            continue
        else:
            abs_time_gaps.append(merged_time_gaps[i])

    # check first and ilast gap against range of all datasets
    if oldest_date > abs_time_gaps[0][0]:
        abs_time_gaps[0][0] = oldest_date
    if newest_date < abs_time_gaps[-1][1]:
        print(abs_time_gaps[-1])
        abs_time_gaps[-1][1] = newest_date
        print(abs_time_gaps[-1])

    # time to slice and dice
    # need to return time vector, buoy tensor, global tensor
    obs_time_result = []
    ww3_time_result = []
    observed_result = []
    ww3_result = []
    for i in range(len(obs_datasets)):
        observed_result.append(np.empty((0, obs_datasets[i].stacked_data.shape[1] - 1), dtype=object))
        obs_time_result.append([])
    for i in range(len(ww3_datasets)):
        ww3_result.append(np.empty((0, ww3_datasets[i].stacked_data_hourly.shape[1] - 1), dtype=object))
        ww3_time_result.append([])

    # start at oldest date
    start_date = oldest_date

    for gap_idx in range(len(abs_time_gaps)):
        gap = abs_time_gaps[gap_idx]
        start_gap = gap[0]
        end_gap = gap[1]

        end_date = start_gap
        # extract data for idx(start_date) to idx(end_date) - 1
        for i in range(len(obs_datasets)):
            start_idx = find_nearest_index(obs_datasets[i].stacked_data[:, 0], start_date)
            end_idx = find_nearest_index(obs_datasets[i].stacked_data[:, 0], end_date)
            obs_time_result[i].extend(obs_datasets[i].stacked_data[start_idx:end_idx, 0])
            observed_result[i] = np.vstack((observed_result[i], obs_datasets[i].stacked_data[start_idx:end_idx, 1:]))

        for i in range(len(ww3_datasets)):
            start_idx = find_nearest_index(ww3_datasets[i].stacked_data_hourly[:, 0], start_date)
            end_idx = find_nearest_index(ww3_datasets[i].stacked_data_hourly[:, 0], end_date)
            ww3_time_result[i].extend(ww3_datasets[i].stacked_data_hourly[start_idx:end_idx, 0])
            ww3_result[i] = np.vstack((ww3_result[i], ww3_datasets[i].stacked_data_hourly[start_idx:end_idx, 1:]))

        start_date = end_gap


    # extract data for range: end of last time gap to end of overlapping data
    start_date = abs_time_gaps[-1][1]
    end_date = newest_date
    for i in range(len(obs_datasets)):
            start_idx = find_nearest_index(obs_datasets[i].stacked_data[:, 0], start_date)
            end_idx = find_nearest_index(obs_datasets[i].stacked_data[:, 0], end_date)
            obs_time_result[i].extend(obs_datasets[i].stacked_data[start_idx:end_idx, 0])
            observed_result[i] = np.vstack((observed_result[i], obs_datasets[i].stacked_data[start_idx:end_idx, 1:]))

    for i in range(len(ww3_datasets)):
            start_idx = find_nearest_index(ww3_datasets[i].stacked_data_hourly[:, 0], start_date)
            end_idx = find_nearest_index(ww3_datasets[i].stacked_data_hourly[:, 0], end_date)
            ww3_time_result[i].extend(ww3_datasets[i].stacked_data_hourly[start_idx:end_idx, 0])
            ww3_result[i] = np.vstack((ww3_result[i], ww3_datasets[i].stacked_data_hourly[start_idx:end_idx, 1:]))


    return obs_time_result, ww3_time_result, observed_result, ww3_result


def merge_time_gaps(time_gaps_list):
    # merge time_gaps that cross over each other
    i = 0
    merged_time_gaps = []
    while i < len(time_gaps_list):
        curr_gap = time_gaps_list[i]
        if i + 1 >= len(time_gaps_list):
            merged_time_gaps.append(curr_gap)
            break
        next_gap = time_gaps_list[i + 1]

        # if no overlap, append and continue
        if curr_gap[1] < next_gap[0]:
            merged_time_gaps.append(curr_gap)
            i += 1
            continue

        new_gap = [curr_gap[0], np.nan]
        while curr_gap[1] >= next_gap[0] and i < len(time_gaps_list) - 1:
            i += 1
            curr_gap = time_gaps_list[i]
            next_gap = time_gaps_list[i + 1]

        new_gap[1] = curr_gap[1]

        merged_time_gaps.append(new_gap)
        i += 1
    return merged_time_gaps


def find_nearest_index(np_array, value):
    idx = (np.abs(np_array - value)).argmin()
    return idx


def save_single_set_to_npz(filename, time_vector, obs_tensor, ww3_tensor, compressed=False):
    """
    Helper class to combine a single set of time, observation and ww3 files
    :param filename:
    :param time_vector:
    :param obs_tensor:
    :param ww3_tensor:
    :return:
    """
    if compressed:
        np.savez_compressed('comp_' + filename, time_vector=time_vector, obs_tensor=obs_tensor, ww3_tensor=ww3_tensor)
    else:
        np.savez(filename, time_vector=time_vector, obs_tensor=obs_tensor, ww3_tensor=ww3_tensor)


def convert_to_third_order_tensor(some_data):
    d_shape = some_data.shape
    new_shape = (d_shape[0], d_shape[1], 64)
    result = np.ones(new_shape)
    for i in range(d_shape[0]):
        for j in range(new_shape[1]):
            result[i, j, :] = some_data[i][j]
    return result


def convert_old_file_to_third_order_tensor(npz_filename, new_filename):
    data = np.load(npz_filename)
    time_vector = data['time_vector']
    obs_tensor_full = convert_to_third_order_tensor(data['obs_tensor'])
    ww3_tensor_full = convert_to_third_order_tensor(data['ww3_tensor'])
    save_single_set_to_npz(new_filename, time_vector, obs_tensor_full, ww3_tensor_full, compressed=False)


if __name__ in ['__main__', 'builtins']:
    buoy = str(46015)
    observed_file = 'data/new_buoys/NDBCObserved_' + buoy + '_hourly.mat'
    ww3_file = 'data/new_buoys/WW3CFSRphase2_' + buoy + '_rebanded.mat'
    #buoy = ObservedDatasetFromFile(observed_file)
    #glob = WW3DatasetFromFile(ww3_file)

    obs_data = scipy.io.loadmat(observed_file)
    ww3_data = scipy.io.loadmat(ww3_file)

    print("obs")
    for i,key in enumerate(obs_data.keys()):
      if (i > 2):
        print("\t", key, ":", obs_data.get(key).shape)

    print("ww3")
    for i,key in enumerate(ww3_data.keys()):
      if (i > 2):
        print("\t", key, ":", ww3_data.get(key).shape)

    obs_times, ww3_times, buoy_tensor, global_tensor = generate_combined_datasets(observed_file, ww3_file)

    print("generate_combined_datasets OBSERVED :", buoy_tensor[0].shape)
    print("generate_combined_datasets WW3      :", global_tensor[0].shape)

    print("convert_to_third_order_tensor(OBSERVED):", convert_to_third_order_tensor(buoy_tensor[0]).shape)
    print("convert_to_third_order_tensor(WW3)     :", convert_to_third_order_tensor(global_tensor[0]).shape)

    print("np.array(obs_times):", np.array(obs_times[0]).shape)

    print("done loading data")
    #target_npz = 'datasets/new_buoys/' + buoy + '_combined_data_' + str(datetime.date.today()) + '.npz'
    #save_single_set_to_npz(target_npz, np.array(obs_times[0]), convert_to_third_order_tensor(buoy_tensor[0]), convert_to_third_order_tensor(global_tensor[0]))
    #print("done saving data")

    # # CONVERTING OLD FILE FORMATS
    # train_npz = '../data/comp_waves_TRAIN_2019-02-25.npz'
    # new_train_npz = "waves_TRAIN_" + str(datetime.date.today()) + '.npz'
    # convert_old_file_to_third_order_tensor(train_npz, new_train_npz)
    #
    # dev_npz = '../data/comp_waves_DEV_2019-02-25.npz'
    # new_dev_npz = "waves_DEV_" + str(datetime.date.today()) + '.npz'
    # convert_old_file_to_third_order_tensor(dev_npz, new_dev_npz)
    #
    # test_npz = '../data/comp_waves_TEST_2019-02-25.npz'
    # new_test_npz = "waves_TEST_" + str(datetime.date.today()) + '.npz'
    # convert_old_file_to_third_order_tensor(test_npz, new_test_npz)

