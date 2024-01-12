# 
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

import numpy as np
from data_utils.matlab_datenums import matlab_datenum_to_py_date as matlab_dtm
from data_utils.constants import FREQUENCY_RANGE, BANDWIDTH_RANGE, E, A1, B1, A2, B2
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import os
import scipy.io
from datetime import datetime
from ConvNN.cnn import ConvNN
from pathlib import Path
class Dataset:

    def __init__(self, npz_file, dataset, h, f, q, offset, train_class=None, moment_shared_std=False,
                     standardization=False, moment_div_e=False):
    
        data = np.load(npz_file)

        # standardization configs from arguments
        self.moment_shared_std = moment_shared_std
        self.standardization = standardization
        self.moment_div_e = moment_div_e

        # set the class variables from arguments
        self.dataset = dataset.lower()
        self.history_len = h
        self.forecast_len = f
        self.freq_bins = q
        self.freq_indices = self.get_freq_indices(q)
        self.offset = offset

        self.for_tensor_full = data["forecast"]
        self.obs_tensor_full = data["observed"]

        self.time_zulu     = data["forecast_time_zulu"]
        self.observed_time = data["observed_time"]

        # slice the data at frequency indices
        self.for_tensor_clip = self.for_tensor_full[:, self.freq_indices[0]:self.freq_indices[1], :, :]
        self.obs_tensor_clip = self.obs_tensor_full[:, self.freq_indices[0]:self.freq_indices[1], :]

        # compute standardized tensors
        if standardization:
            self.set_standardization_scheme(moment_shared_std, train_class)
            self.for_tensor = (self.for_tensor_clip - self.mean) / self.std
            self.obs_tensor = (self.obs_tensor_clip - self.mean) / self.std
        elif moment_div_e:
            self.set_div_e_scheme()
        else:
            self.for_tensor = self.for_tensor_clip
            self.obs_tensor = self.obs_tensor_clip
            self.mean = np.zeros((5, self.obs_tensor.shape[1]), dtype=np.float64)
            self.std = np.ones((5, self.obs_tensor.shape[1]), dtype=np.float64)

        # generate the start indices from the time vector
        # a valid start index has a forecast where time_zulu[0] == obs_time[0]
        # and there is a forecast where time_zulu[0] == obs_time[-1] + 1
        self.calc_valid_start_indices(self)

        self.shuffle_valid_start_indices()
        self.batch_pos = 0

    def set_div_e_scheme(self):
        # initialize tensors to default values
        self.obs_tensor = self.obs_tensor_clip
        self.for_tensor = self.for_tensor_clip

        self.mean = np.zeros((5, self.obs_tensor.shape[1]), dtype=np.float64)
        self.std = np.ones((5, self.obs_tensor.shape[1]), dtype=np.float64)

        # divide a1/b1, a2/b2 by e
        idx_list = [A1, B1, A2, B2]
        for idx in idx_list:
            if (np.any(self.obs_tensor[idx, :, :] < -1) and np.all(self.obs_tensor[idx, :, :] > 1)):
              self.obs_tensor[idx, :, :] = self.obs_tensor[idx, :, :] / self.obs_tensor[E, :, :]
              self.obs_tensor[idx, :, :] = np.nan_to_num(self.obs_tensor[idx, :, :])
            if (np.any(self.for_tensor[idx, :, :, :] < -1) and np.all(self.for_tensor[idx, :, :, :] > 1)):
              self.for_tensor[idx, :, :, :] = self.for_tensor[idx, :, :, :] / self.for_tensor[E, :, :, :]
              self.for_tensor[idx, :, :, :] = np.nan_to_num(self.for_tensor[idx, :, :, :])

        # standardize e  :
        reshape_for_E = np.reshape(self.for_tensor[E, :, :, :], (self.for_tensor.shape[1], -1))
        self.mean[E, :] = np.mean(np.append(self.obs_tensor[E, :, :], reshape_for_E, axis=1), axis=1)
        self.std[E, :] = np.std(np.append(self.obs_tensor[E, :, :], reshape_for_E, axis=1), axis=1)

        # standardize(self.obs_tensor_clip - self.mean) / self.std
        self.obs_tensor[E, :, :] = ((self.obs_tensor[E, :, :].transpose() - self.mean[E, :]) / self.std[E, :]).transpose()
        for i in range(self.for_tensor.shape[-1]):
          self.for_tensor[E, :, :, i] = ((self.for_tensor[E, :, :, i].transpose() - self.mean[E, :]) / self.std[E, :]).transpose()


    def set_freq_bins(self, q):
        """

        Args:
            q: # freq bins: freq_indices: index of start of frequency, and index of end frequency + 1

        Returns:

        """
        self.freq_bins = q
        self.freq_indices = self.get_freq_indices(q)

    @staticmethod
    def calc_valid_start_indices(self):
        if None in [self.history_len, self.freq_bins, self.forecast_len, self.offset]:
            raise Exception("In batcher.Dataset.calc_valid_indices: class objects are None")

        valid_indices = []
        f = self.forecast_len
        h = self.history_len
        t1 = t2 = t3 = t4 = t5 = t6 = 0
        for i in range(len(self.time_zulu)-1):
            test1 = test2 = test3 = test4 = test5 = test6 = test7 = test8 = False

            # forecast 0 has to have no nans
            if (not np.isnan(self.for_tensor[:, :, 0:f, i]).any()):
                test1 = True
            # valid forecast -h start index
            if (i-(h/6) >= 0):
                test2 = True
            # forecast -h has to have no nans
            if (not np.isnan(self.for_tensor[:, :, 0:h+f, i-(h//6)]).any()):
                test3 = True
            # time_zulu[-h start] w/in h of time_zulu[0 start]
            if (self.time_zulu[i] - (self.time_zulu[i-(h//6)]) < (h * 0.0417)):
                test4 = True
            # valid obs start index
            if (i*6-h >= 0 and i*6-h < self.obs_tensor.shape[2]):
                test5 = True
            # valid obs end index
            if (i*6 >= 0 and i*6 < self.obs_tensor.shape[2]):
                test6 = True
            # valid obs start index for targets
            if (i*6+f < self.obs_tensor.shape[2]):
                test7 = True
            # obs has to have no nans
            if (not np.isnan(self.obs_tensor[:, :, i*6-h:i*6]).any()):
                test8 = True


            if (test1 and test2 and test3 and test4 and test5 and
               test6 and test7 and test8): 
                valid_indices.append(i)

        print("num valid idx for", self.dataset, "dataset:", len(valid_indices))
        self.valid_start_indices = np.array(valid_indices)


    @staticmethod
    def convert_to_third_order_tensor(some_data):
        """

        Args:
            some_data:

        Returns:

        """
        d_shape = some_data.shape
        new_shape = (d_shape[0], d_shape[1], 64)
        result = np.ones(new_shape)
        for i in range(d_shape[0]):
            for j in range(new_shape[1]):
                result[i, j, :] = some_data[i][j]
        return result

    @staticmethod
    def get_freq_indices(q):
        """

        Args:
            q:

        Returns:

        """
        if q is None:
            return None

        freq_indices = [0, 0]
        # Find the index of the value in FREQUENCY_RANGE that is closest to the value q[0]
        freq_indices[0] = (np.abs(np.array(FREQUENCY_RANGE) - q[0])).argmin()
        freq_indices[1] = (np.abs(np.array(FREQUENCY_RANGE) - q[1])).argmin() + 1
        return freq_indices

    def shuffle_valid_start_indices(self):
        np.random.shuffle(self.valid_start_indices)

    def get_mini_batch(self, mb_size, full=False, as_torch_cuda=False, resids=False):
        """
        function that takes a batch of start hours and produces minibatches of MBx60x5x64 input and a MBx12x5x64 output

        04/11 -- updating to return inputs, y_for, and y_buoy

        """
        # get the length of valid start indices
        length = self.valid_start_indices.shape[0]

        # check to see if we are taking full batches instead
        if full or mb_size > length:
            mb_size = length
            self.batch_pos = 0

        # check to see if we are at the end of batches
        if mb_size > length - self.batch_pos:
            self.batch_pos = 0
            self.shuffle_valid_start_indices()
            return None, None, None

        # create target arrays
        # input MB x (2 * H + F) x 5 x Q
        # output MB x F x 5 x Q
        input_len = 2 * self.history_len + 2 * self.forecast_len
        q_len = self.freq_indices[1] - self.freq_indices[0]
        inputs = np.zeros((mb_size, input_len, 5, q_len))
        y_buoy = np.zeros((mb_size, self.forecast_len, 5, q_len))
        y_for = np.zeros((mb_size, self.forecast_len, 5, q_len))

        # get the data
        for i in range(mb_size):
            # Get the tensor starting index
            t_idx = self.valid_start_indices[self.batch_pos]
            # extract obs_data

            obs_start = 0
            obs_len = self.history_len
            obs_end = obs_start + obs_len
            temp = self.obs_tensor[:, :, t_idx*6-obs_len:t_idx*6]
            for j in range(obs_len):
                inputs[i, obs_start+j, :, :] = temp[:, :, j]

            # extract for_data
            for_0_start = obs_end
            for_0_len = self.forecast_len
            for_0_end = for_0_start + for_0_len
            temp = self.for_tensor[:, :, 0:for_0_len, t_idx]
            for j in range(for_0_len):
                inputs[i, for_0_start+j, :, :] = temp[:, :, j]

            for_1_start = for_0_end
            for_1_len = self.history_len+self.forecast_len
            for_1_end = for_1_start + for_1_len
            temp = self.for_tensor[:, :, 0:for_1_len, t_idx-(self.history_len//6)]
            for j in range(for_1_len):
                inputs[i, for_1_start+j, :, :] = temp[:, :, j]

            # then extract forecast_len outputs into targets
            temp = self.for_tensor[:, :, 0:for_0_len, t_idx]
            for j in range(for_0_len):
                y_for[i, j, :, :] = temp[:, :, j]
            temp = self.obs_tensor[:, :, t_idx*6:t_idx*6+self.forecast_len]
            for j in range(self.forecast_len):
                y_buoy[i, :, :, :] = temp[:, :, j]

            # update that batch position
            self.batch_pos += 1

        # convert to torch tensor
        inputs = torch.from_numpy(inputs).float().cuda() if as_torch_cuda else torch.from_numpy(inputs).float()
        y_for = torch.from_numpy(y_for).float().cuda() if as_torch_cuda else torch.from_numpy(y_for).float()
        y_buoy = torch.from_numpy(y_buoy).float().cuda() if as_torch_cuda else torch.from_numpy(y_buoy).float()


        return inputs, y_for, y_buoy

    def generate_mat_lab_file(self, filename, data_dict):
        # scipy.io.savemat(filename, dict(x=data['x'], y=data['y']))
        scipy.io.savemat(filename, data_dict)

    def create_predictions(self, model_path, on_gpu, model=None, test_set=False):

        model_name = model_path.split("/")[-1]
        loss_function = nn.MSELoss()

        if on_gpu:
            model = torch.load(model_path)
        else:
            device = torch.device('cpu')
            model = torch.load(model_path, map_location=device)
#            model.load_state_dict(torch.load(model_path, map_location=device))


        # shell inputs (1 x (2H + F) x 5 x Q)
        h = self.history_len
        f = self.forecast_len
        h_f = h + f
        q = self.freq_indices[1] - self.freq_indices[0]
        inputs = np.zeros((1, 2 * h + f, 5, q))

        # empty tensor for target for data
        y_for = np.zeros((1, f, 5, q))

        y_buoy = np.zeros((1, f, 5, q))

        # Hours bucket will be a tensor for each hour in the forecast
        # (hour x N x 5 x q)
        hour_buckets = np.empty((f, self.length, 5, q))
        hour_buckets[:] = np.nan

        # collect the data into buckets
        mean = torch.from_numpy(self.mean).float()
        std = torch.from_numpy(self.std).float()

        test_running_loss = 0.0
        test_num_batches = 0
        for i in range(self.length - h_f):
            # get our inputs (MB x (2H + F) x 5 x Q)
            inputs[0, 0:f, :, :] = self.for_tensor[:, :, i:i+f, :]  # f hours of for
            inputs[0, 0:h_f, :, :] = self.for_tensor[:, :, i:i+h_f, :]  # h + f hours of for
            inputs[0, h:, :, :] = self.obs_tensor[:, :, i:i+h]     # h hours of obs
            y_for[0, 0:f, :, :] = self.for_tensor[i + h:i + h_f]     # f hours of for

            # predict residuals at i + H through i + H + F
            inputs_t = torch.from_numpy(inputs).float()
            y_for_t = torch.from_numpy(y_for).float()

            with torch.no_grad():
                if on_gpu:
                    predictions, _ = model(inputs_t.cuda(), y_for_t.cuda(), mean.cuda(), std.cuda())
                else:
                    predictions, _ = model(inputs_t, y_for_t)

            # store predictions in bucket at proper position
            for hr in range(f):
                i_offset = i + h + hr
                hour_buckets[hr, i_offset] = predictions[0, hr]

        # reverse normalize predictions on hours
        hour_buckets = hour_buckets[:, :] * self.std + self.mean

        # remove negative energy
        hour_buckets[:, :, E, :] = hour_buckets[:, :, E, :].clip(min=0.0)

        # reverse div_e if applicable
        if self.moment_div_e:
            hour_buckets[:, :, A1, :] = hour_buckets[:, :, A1, :] * hour_buckets[:, :, E, :]
            hour_buckets[:, :, B1, :] = hour_buckets[:, :, B1, :] * hour_buckets[:, :, E, :]
            hour_buckets[:, :, A2, :] = hour_buckets[:, :, A2, :] * hour_buckets[:, :, E, :]
            hour_buckets[:, :, B2, :] = hour_buckets[:, :, B2, :] * hour_buckets[:, :, E, :]

        # save obs, for, and predictions to matlab file
        def create_data_dict(d):
            data_dict = {
                'a1': d[:, 0, :],
                'a2': d[:, 1, :],
                'b1': d[:, 2, :],
                'b2': d[:, 3, :],
                'e':  d[:, 4, :],
                'time': self.time_vector,
                'bw': np.array(BANDWIDTH_RANGE[self.freq_indices[0]: self.freq_indices[1]]),
                'fr': np.array(FREQUENCY_RANGE[self.freq_indices[0]: self.freq_indices[1]]),
            }
            return data_dict

        # create the target directory
        time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        type_str = 'test' if test_set else 'dev'
        target_dir = os.getcwd() + '/data_outputs/' + model_name + '_' + type_str + '_' + str(self.history_len) + '-' + str(self.forecast_len) + '_' + time_stamp + '/'
        os.mkdir(target_dir)

        # Generate the observed file
        data_dict = create_data_dict(self.reverse_pre_processing(self.obs_tensor))
        curr_file = target_dir + time_stamp + '_obs_data.mat'
        self.generate_mat_lab_file(curr_file, data_dict)

        # Generate the for file
        data_dict = create_data_dict(self.reverse_pre_processing(self.for_tensor))
        curr_file = target_dir + time_stamp + '_for_data.mat'
        self.generate_mat_lab_file(curr_file, data_dict)

        # Generate the hour buckets files
        for hr in range(f):
            data_dict = create_data_dict(hour_buckets[hr])
            curr_file = target_dir + time_stamp + '_hr' + str(hr + 1) + '_data.mat'
            self.generate_mat_lab_file(curr_file, data_dict)

    def reverse_pre_processing(self, tensor_object):

        # reverse standardization (* self.std + self.mean)
        tensor_object = tensor_object * self.std + self.mean

        # reverse moment if applicable
        if self.moment_div_e:
            tensor_object[:, A1, :] = tensor_object[:, A1, :] * tensor_object[:, E, :]
            tensor_object[:, A2, :] = tensor_object[:, A2, :] * tensor_object[:, E, :]
            tensor_object[:, B1, :] = tensor_object[:, B1, :] * tensor_object[:, E, :]
            tensor_object[:, B2, :] = tensor_object[:, B2, :] * tensor_object[:, E, :]

        return tensor_object


if __name__ in ['__main__', 'builtins']:
    file_prefix = '/research/hutchinson/projects/ml_waves19/ml_waves19_chloe/datasets/'
    train_fn = file_prefix + 'waves_TRAIN_2019-03-03.npz'
    dev_fn = file_prefix + 'waves_DEV_2019-03-03.npz'
    test_fn = 'data/waves_TEST_2019-03-03.npz'
    hist = 36
    forecast = 24
    q_freq = [0.04, 0.25]
    off_set = 6

    train = Dataset(train_fn, 'train', hist, forecast, q_freq, off_set)
    dev = Dataset(dev_fn, 'dev', hist, forecast, q_freq, off_set, train_class=train)

