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
    """
    Dataset class

    Note:

    Args:
        npz_file: numpy zip file name/location
        dataset: one of 'train', 'dev' 'test'
        h (int): history length
        f (int): forecast length
        q [int, int] : # freq bins: freq_indices: index of start of frequency, and index of end frequency + 1
        offset (int):  hours of tolerance between data points

    Attributes:
        dataset (str): one of 'train', 'dev' 'test'
        history_len (int):
        forecast_len (int):
        freq_bins (array[2]):
        freq_indices ():
        offset (int):
        time_vector
        obs_tensor
        ww3_tensor

    """
    def __init__(self, npz_file, dataset, h, f, q, offset, train_class=None, moment_shared_std=False,
                      standardization=False, moment_div_e=False, max_hours=0, start_end_date=[0,0]):

        # Data pre-processing schemes
        # standardization:     flag to signal data standardization
        # -->       if moment_shared_std:  a1/b1 and a2/b2 will share std/mean
        # div_e:  special flag -  e is standardized, a1/b2, a2/b2 are divided by e without standardization

        # Load the data from npz
        data = np.load(npz_file)

        # standardization configs
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

        # Load the vectors/tensors from the numpy zip data
        self.time_vector = data['time_vector']

        #### WRITE DATES TO FILE ####
        if (dataset == "train"):
          print("writing dates....",end="")
          dates_file = open(Path("dates.txt"),"r+")
          dates_file.truncate()
          for i,time in enumerate(self.time_vector):
            dates_file.write(str(matlab_dtm(time))+"\n")
          dates_file.close()
          print("done with ", i, "dates")

        if (start_end_date[0] > 0):
          print("time_vector", self.time_vector)
          starti = np.where(self.time_vector == start_end_date[0])[0][0]
          endi = np.where(self.time_vector == start_end_date[1])[0][0]
          self.time_vector = self.time_vector[starti:endi]
        else:
          self.time_vector = self.time_vector[-max_hours:]
        self.length = self.time_vector.shape[0]

        # Get the data tensors  (N x 5 x 64)
        self.obs_tensor_full = data['obs_tensor']
        self.ww3_tensor_full = data['ww3_tensor']

        # Slice data at frequency indices  (N x 5 x Q)
        if (start_end_date[0] > 0):
            self.obs_tensor_clip = self.obs_tensor_full[starti:endi, :, self.freq_indices[0]:self.freq_indices[1]]
            self.ww3_tensor_clip = self.ww3_tensor_full[starti:endi, :, self.freq_indices[0]:self.freq_indices[1]]
        else:
            self.obs_tensor_clip = self.obs_tensor_full[-max_hours:, :, self.freq_indices[0]:self.freq_indices[1]]
            self.ww3_tensor_clip = self.ww3_tensor_full[-max_hours:, :, self.freq_indices[0]:self.freq_indices[1]]

        if (dataset == "train"):
            print('start date ',str(matlab_dtm(self.time_vector[0])))
            print('end date ',str(matlab_dtm(self.time_vector[-1])))
            yrs = (self.time_vector[-1] - self.time_vector[0])//365.25
            dys = (self.time_vector[-1] - self.time_vector[0])%365.25
            print("** **", yrs, "YEARS and ", dys ,"DAYS of data ** **") 

        # compute the standardized tensors
        if standardization:
            self.set_standardization_scheme(moment_shared_std, train_class)
            self.obs_tensor = (self.obs_tensor_clip - self.mean) / self.std
            self.ww3_tensor = (self.ww3_tensor_clip - self.mean) / self.std
        elif moment_div_e:
            self.set_div_e_scheme()
        else:
            self.obs_tensor = self.obs_tensor_clip
            self.ww3_tensor = self.ww3_tensor_clip
            self.mean = np.zeros((5, self.obs_tensor.shape[-1]), dtype=np.float64)
            self.std = np.ones((5, self.obs_tensor.shape[-1]), dtype=np.float64)

        # compute the residuals tensor
        self.resid_tensor = self.obs_tensor - self.ww3_tensor

        # Generate the start indices from the time vector, this is important in finding
        # gaps within the time_vector and used for computing valid start indexes next
        self.time_gap_indices = np.array([[0]])
        threshold = 0.04167  # relative threshold between hours: 0.04166666662786156
        times = self.time_vector[1:self.length] - self.time_vector[0:self.length - 1]
        self.time_gap_indices = np.append(self.time_gap_indices, np.argwhere(times > threshold) + 1, axis=0)

        # Compute the valid start indexes
        self.valid_start_indices = None
        self.batch_pos = 0
        if None not in [h, f, q, offset]:
            self.calc_valid_start_indices(self)

        # shuffle hours and create a batch mask
        self.shuffle_valid_start_indices()

    def set_div_e_scheme(self):
        # initialize tensors to default values
        self.obs_tensor = self.obs_tensor_clip
        self.ww3_tensor = self.ww3_tensor_clip
        self.mean = np.zeros((5, self.obs_tensor.shape[-1]), dtype=np.float64)
        self.std = np.ones((5, self.obs_tensor.shape[-1]), dtype=np.float64)

        # divide a1/b1, a2/b2 by e
        idx_list = [A1, B1, A2, B2]
        for idx in idx_list:
            self.obs_tensor[:, idx, :] = self.obs_tensor[:, idx, :] / self.obs_tensor[:, E, :]
            self.ww3_tensor[:, idx, :] = self.ww3_tensor[:, idx, :] / self.ww3_tensor[:, E, :]
            self.obs_tensor[:, idx, :] = np.nan_to_num(self.obs_tensor[:, idx, :])
            self.ww3_tensor[:, idx, :] = np.nan_to_num(self.ww3_tensor[:, idx, :])

        # standardize e  :
        self.mean[E, :] = np.mean(np.append(self.obs_tensor[:, E, :], self.ww3_tensor[:, E, :], axis=0), axis=0)
        self.std[E, :] = np.std(np.append(self.obs_tensor[:, E, :], self.ww3_tensor[:, E, :], axis=0), axis=0)


       # standardize(self.obs_tensor_clip - self.mean) / self.std
        self.obs_tensor[:, E, :] = (self.obs_tensor[:, E, :] - self.mean[E, :]) / self.std[E, :]
        self.ww3_tensor[:, E, :] = (self.ww3_tensor[:, E, :] - self.mean[E, :]) / self.std[E, :]


    def set_standardization_scheme(self, moment_shared_std, train_class):
        """
        Sets up the standardization scheme based on class arguments
        Args:
            moment_shared_std: determines if a1/b1 share a std and mean
            train_class:

        Returns:

        """

        # Get the corresponding mean and standard deviation for normalization
        if self.dataset == 'train':
            # Compute mean (5 x freq_range)
            if moment_shared_std:
                combo = np.append(self.obs_tensor_clip, self.ww3_tensor_clip, axis=0)
                mean_a1b1 = np.mean(np.concatenate((combo[:, A1, :], combo[:, B1, :]), axis=0), axis=0)
                mean_a2b2 = np.mean(np.concatenate((combo[:, A2, :], combo[:, B2, :]), axis=0), axis=0)
                mean_e = np.mean(combo[:, E, :], axis=0)
                self.mean = np.vstack((mean_a1b1, mean_a2b2, mean_a1b1, mean_a2b2, mean_e))

                std_a1b1 = np.std(np.concatenate((combo[:, A1, :], combo[:, B1, :]), axis=0), axis=0)
                std_a2b2 = np.std(np.concatenate((combo[:, A2, :], combo[:, B2, :]), axis=0), axis=0)
                std_e = np.std(combo[:, E, :], axis=0)
                self.std = np.vstack((std_a1b1, std_a2b2, std_a1b1, std_a2b2, std_e))

            else:
                self.mean = np.mean(np.append(self.obs_tensor_clip, self.ww3_tensor_clip, axis=0), axis=0)
                self.std = np.std(np.append(self.obs_tensor_clip, self.ww3_tensor_clip, axis=0), axis=0)

        elif train_class is not None:
            self.mean = train_class.mean
            self.std = train_class.std

        else:
            raise Exception(self.dataset + " requires train class for standardization.")

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
        """

        Args:
            self:

        Returns:

        """
        if None in [self.history_len, self.freq_bins, self.forecast_len, self.offset]:
            raise Exception("In batcher.Dataset.calc_valid_indices: class objects are None")

        gap_reach = self.history_len + self.forecast_len

        def calc_upper(low, up):
            return low + (up - low - gap_reach) // gap_reach * gap_reach

        valid_indices = []
        for i in range(len(self.time_gap_indices) - 1):
            lower = self.time_gap_indices[i, 0]
            upper = self.time_gap_indices[i+1, 0] - 1

            # get nearest divisible by upper gap reach
            upper = calc_upper(lower, upper)

            # finding the range for x[i], x[i + 1]
            valid_indices.extend(range(lower, upper, self.offset))

        # Get the ending gap
        lower = self.time_gap_indices[-1, 0]
        upper = self.length
        upper = calc_upper(lower, upper)
        valid_indices.extend(range(lower, upper, self.offset))

        # set the class attribute
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

        04/11 -- updating to return inputs, y_ww3, and y_buoy

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
        input_len = 2 * self.history_len + self.forecast_len
        q_len = self.freq_indices[1] - self.freq_indices[0]
        inputs = np.zeros((mb_size, input_len, 5, q_len))
        y_buoy = np.zeros((mb_size, self.forecast_len, 5, q_len))
        y_ww3 = np.zeros((mb_size, self.forecast_len, 5, q_len))
        residuals = np.zeros((mb_size, self.forecast_len, 5, q_len))

        # get the data
        for i in range(mb_size):
            # Get the tensor starting index
            t_idx = self.valid_start_indices[self.batch_pos]

            # extract ww3_data
            ww3_start = 0
            ww3_len = self.history_len + self.forecast_len
            ww3_end = ww3_start + ww3_len
            inputs[i, ww3_start:ww3_end] = self.ww3_tensor[t_idx:t_idx + ww3_len, :, :]

            # extract obs_data
            obs_start = ww3_end
            obs_len = self.history_len
            obs_end = obs_start + obs_len
            # first extract first history_len values into inputs
            inputs[i, obs_start:obs_end] = self.obs_tensor[t_idx:t_idx + obs_len, :, :]

            # then extract forecast_len outputs into targets
            tgt_start = 0
            tgt_len = self.forecast_len
            tgt_end = tgt_start + tgt_len
            start_idx = t_idx + obs_len
            end_idx = start_idx + tgt_len
            y_ww3[i, :] = self.ww3_tensor[start_idx:end_idx, :, :]
            y_buoy[i, :] = self.obs_tensor[start_idx:end_idx, :, :]
            residuals[i, :] = self.resid_tensor[start_idx:end_idx, :, :]

            # update that batch position
            self.batch_pos += 1

        # convert to torch tensor
        inputs = torch.from_numpy(inputs).float().cuda() if as_torch_cuda else torch.from_numpy(inputs).float()
        y_ww3 = torch.from_numpy(y_ww3).float().cuda() if as_torch_cuda else torch.from_numpy(y_ww3).float()
        y_buoy = torch.from_numpy(y_buoy).float().cuda() if as_torch_cuda else torch.from_numpy(y_buoy).float()

        return inputs, y_ww3, y_buoy

        # if resids:
        #     residuals = torch.from_numpy(residuals).float().cuda() if as_torch_cuda else torch.from_numpy(residuals).float()
        #     return inputs, residuals
        # else:
        #     y_buoy = torch.from_numpy(y_buoy).float().cuda() if as_torch_cuda else torch.from_numpy(y_buoy).float()
        #
        # return inputs, y_buoy

    def generate_mat_lab_file(self, filename, data_dict):
        # scipy.io.savemat(filename, dict(x=data['x'], y=data['y']))
        scipy.io.savemat(filename, data_dict)

    def analyze_md_frequency_loss(self, hourly_pred, rmse=False):
        exp = 0.5 if rmse else 1.0
        h = self.history_len
        f = self.forecast_len
        h_f = h + f
        loss_function = nn.MSELoss()

        t_ww3 = torch.from_numpy(self.ww3_tensor * self.std + self.mean)
        t_obs = torch.from_numpy(self.obs_tensor * self.std + self.mean)
        t_hrs = torch.from_numpy(hourly_pred[:] * self.std + self.mean)

        frequencies = t_obs.shape[-1]

        ww3_loss = loss_function(torch.atan2(t_ww3[:, B1, :], t_ww3[:, A1, :]),
                                 torch.atan2(t_obs[:, B1, :], t_obs[:, A1, :])).item() ** exp
        print("WW3 MD LOSS: ", ww3_loss)


        print("computing md loss per frequency and hour")
        with torch.no_grad():
            for hr in range(f):
                hr_item = t_hrs[hr, h + hr:-h_f, :, :]
                obs_item = t_obs[h + hr:-h_f, :, :]
                for fr in range(frequencies):
                    component_loss = loss_function(torch.atan2(hr_item[:, B1, fr], hr_item[:, A1, fr]), torch.atan2(obs_item[:, B1, fr], obs_item[:, B1, fr])).item() ** exp
                    # component_loss = loss_function(hr_item[:, B1, fr], obs_item[:, B1, fr])
                    # component_loss = loss_function(hr_item[:, B1, fr] / hr_item[:, A1, fr], obs_item[:, B1, fr] / obs_item[:, A1, fr]).item() ** exp
                    # component_loss = loss_function(torch.mean(hr_item[:, B1, fr]), torch.mean(obs_item[:, B1, fr])) * 100.0
                    # component_loss = torch.mean(hr_item[:, A1, fr]) - torch.mean(obs_item[:, A1, fr])
                    print('%.4f' % component_loss, end="\t")
                print()  # new line
        return

    def analyze_hourly_loss(self, hourly_pred, unstandardize=False, rmse=False):
        exp = 0.5 if rmse else 1.0
        h = self.history_len
        f = self.forecast_len
        h_f = h + f

        loss_function = nn.MSELoss()

        if unstandardize:
            t_ww3 = torch.from_numpy(self.ww3_tensor * self.std + self.mean)
            t_obs = torch.from_numpy(self.obs_tensor * self.std + self.mean)
            t_hrs = torch.from_numpy(hourly_pred[:] * self.std + self.mean)
        else:
            t_ww3 = torch.from_numpy(self.ww3_tensor)
            t_obs = torch.from_numpy(self.obs_tensor)
            t_hrs = torch.from_numpy(hourly_pred)

        # column header
        print("hour", "loss", "e_loss", "a1_loss", "b1_loss", "md_loss", "md_swll", "md_seas", sep='\t')
        item_str = "%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f"

        # swell [0:12], sea [13:

        # WW3 loss versus observations
        with torch.no_grad():
            loss_item = loss_function(t_ww3, t_obs).item() ** exp
            e_loss = loss_function(t_ww3[:, E, :], t_obs[:, E, :]).item() ** exp
            a1_loss = loss_function(t_ww3[:, A1, :], t_obs[:, A1, :]).item() ** exp
            b1_loss = loss_function(t_ww3[:, B1, :], t_obs[:, B1, :]).item() ** exp
            md_loss = loss_function(torch.atan2(t_ww3[:, B1, :], t_ww3[:, A1, :]),
                                    torch.atan2(t_obs[:, B1, :], t_obs[:, A1, :])).item() ** exp
            md_loss_swell = loss_function(torch.atan2(t_ww3[:, B1, 0:12], t_ww3[:, A1, 0:12]),
                                          torch.atan2(t_obs[:, B1, 0:12], t_obs[:, A1, 0:12])).item() ** exp
            md_loss_seas = loss_function(torch.atan2(t_ww3[:, B1, 12:], t_ww3[:, A1, 12:]),
                                          torch.atan2(t_obs[:, B1, 12:], t_obs[:, A1, 12:])).item() ** exp
        print(item_str % ('ww3', loss_item, e_loss, a1_loss, b1_loss, md_loss, md_loss_swell, md_loss_seas))

        # hourly losses versus observations
        for hr in range(f):
            hr_item = t_hrs[hr, h + hr:-h_f, :, :]
            obs_item = t_obs[h + hr:-h_f, :, :]
            with torch.no_grad():
                loss_hr = loss_function(hr_item, obs_item).item() ** exp
                e_loss = loss_function(hr_item[:, E, :], obs_item[:, E, :]).item() ** exp
                a1_loss = loss_function(hr_item[:, A1, :], obs_item[:, A1, :]).item() ** exp
                b1_loss = loss_function(hr_item[:, B1, :], obs_item[:, B1, :]).item() ** exp
                md_loss = loss_function(torch.atan2(hr_item[:, B1, :], hr_item[:, A1, :]),
                                        torch.atan2(obs_item[:, B1, :], obs_item[:, A1, :])).item() ** exp
                md_loss_swell = loss_function(torch.atan2(hr_item[:, B1, 0:12], hr_item[:, A1, 0:12]),
                                              torch.atan2(obs_item[:, B1, 0:12], obs_item[:, A1, 0:12])).item() ** exp
                md_loss_seas = loss_function(torch.atan2(hr_item[:, B1, 12:], hr_item[:, A1, 12:]),
                                             torch.atan2(obs_item[:, B1, 12:], obs_item[:, A1, 12:])).item() ** exp
                print(item_str % (str(hr + 1), loss_hr, e_loss, a1_loss, b1_loss, md_loss, md_loss_swell, md_loss_seas))
        return

    def create_predictions(self, buoy, model_path, on_gpu, model=None, test_set=False):

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

        # empty tensor for target ww3 data
        y_ww3 = np.zeros((1, f, 5, q))

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
            inputs[0, 0:h_f, :, :] = self.ww3_tensor[i:i+h_f, :, :]  # h+f hours of ww3
            inputs[0, h_f:, :, :] = self.obs_tensor[i:i+h, :, :]     # h hours of obs
            y_ww3[0, 0:f, :, :] = self.ww3_tensor[i+h:i+h_f, :, :]   # f hours of ww3
            y_buoy[0, 0:f, :, :] = self.obs_tensor[i+h:i+h_f, :, :]

            # predict residuals at i + H through i + H + F
            inputs_t = torch.from_numpy(inputs).float()
            y_ww3_t = torch.from_numpy(y_ww3).float()
            y_buoy_t = torch.from_numpy(y_buoy).float()

            with torch.no_grad():
                if on_gpu:
                    predictions, _ = model(inputs_t.cuda(), y_ww3_t.cuda(), mean.cuda(), std.cuda())
                else:
                    predictions, _ = model(inputs_t, y_ww3_t, mean, std)

                # compute loss
                test_running_loss += loss_function(predictions, y_buoy_t).item()
                test_num_batches += 1

            # store predictions in bucket at proper position
            for hr in range(f):
                i_offset = i + h + hr
                hour_buckets[hr, i_offset] = predictions[0, hr]

        print("test running loss = ", test_running_loss / test_num_batches)

        # DEBUGGING: report the standardized loss for each prediction
        # # self.analyze_hourly_loss(hour_buckets, unstandardize=True, rmse=False)
        # self.analyze_md_frequency_loss(hour_buckets, unstandardize=False, rmse=True)
        # self.analyze_md_frequency_loss(hour_buckets, rmse=True)
        # return

        # reverse normalize predictions on hours
        hour_buckets = hour_buckets[:, :] * self.std + self.mean

        # DEBUGGING: check the predictions if they are out of bounds
        # if self.moment_div_e:
        #     count = np.where(np.abs(hour_buckets[:, :, 0:4, :]) > 1.0, 1, 0)
        #     print(count.sum())
        #     return

        # remove negative energy
        hour_buckets[:, :, E, :] = hour_buckets[:, :, E, :].clip(min=0.0)

        # DEBUGGING: check that energy is >= 0
        # print(hour_buckets[:, :, E, :].shape)
        # count = np.where(hour_buckets[:, :, E, :] < -0.0000000001, 1, 0)
        # print(count.sum())
        # return

        # DEBUGGING:  checking if a1/b1 go out of bounds
        # count_a1 = np.where(np.abs(hour_buckets[:, :, A1, :]) > 1.0, 1, 0)
        # count_b1 = np.where(np.abs(hour_buckets[:, :, B1, :]) > 1.0, 1, 0)
        # print("Out of Bounds: ", count_a1.sum(), count_b1.sum())
        # return None

        # reverse div_e if applicable
        if self.moment_div_e:
            hour_buckets[:, :, A1, :] = hour_buckets[:, :, A1, :] * hour_buckets[:, :, E, :]
            hour_buckets[:, :, B1, :] = hour_buckets[:, :, B1, :] * hour_buckets[:, :, E, :]
            hour_buckets[:, :, A2, :] = hour_buckets[:, :, A2, :] * hour_buckets[:, :, E, :]
            hour_buckets[:, :, B2, :] = hour_buckets[:, :, B2, :] * hour_buckets[:, :, E, :]

        # save obs, ww3, and predictions to matlab file
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
        target_dir = os.getcwd() + '/data_outputs/b'+str(buoy)+'/' + model_name + '_' + type_str + '_' + str(self.history_len) + '-' + str(self.forecast_len) + '_' + time_stamp + '/'
        os.mkdir(target_dir)

        # Generate the observed file
        data_dict = create_data_dict(self.reverse_pre_processing(self.obs_tensor))
        curr_file = target_dir + time_stamp + '_obs_data.mat'
        self.generate_mat_lab_file(curr_file, data_dict)

        # Generate the ww3 file
        data_dict = create_data_dict(self.reverse_pre_processing(self.ww3_tensor))
        curr_file = target_dir + time_stamp + '_ww3_data.mat'
        self.generate_mat_lab_file(curr_file, data_dict)

        # Generate the hour buckets files
        for hr in range(f):
            data_dict = create_data_dict(hour_buckets[hr])
            curr_file = target_dir + time_stamp + '_hr' + str(hr + 1) + '_data.mat'
            self.generate_mat_lab_file(curr_file, data_dict)

    def reverse_pre_processing(self, tensor_object):

        # reverse standardization (* self.std + self.mean)
        print(tensor_object.shape)
        print(self.std.shape)
        tensor_object = tensor_object * self.std + self.mean

        # reverse moment if applicable
        if self.moment_div_e:
            tensor_object[:, A1, :] = tensor_object[:, A1, :] * tensor_object[:, E, :]
            tensor_object[:, A2, :] = tensor_object[:, A2, :] * tensor_object[:, E, :]
            tensor_object[:, B1, :] = tensor_object[:, B1, :] * tensor_object[:, E, :]
            tensor_object[:, B2, :] = tensor_object[:, B2, :] * tensor_object[:, E, :]

        return tensor_object


if __name__ in ['__main__', 'builtins']:
    file_prefix = '/research/hutchinson/projects/ml_waves19/ml_waves19_noah/datasets/'
    train_fn = file_prefix + 'waves_TRAIN_2019-03-03.npz'
    dev_fn = file_prefix + 'waves_DEV_2019-03-03.npz'
    test_fn = 'data/waves_TEST_2019-03-03.npz'
    hist = 36
    forecast = 24
    q_freq = [0.04, 0.25]
    off_set = 6

    train = Dataset(train_fn, 'train', hist, forecast, q_freq, off_set)
    dev = Dataset(dev_fn, 'dev', hist, forecast, q_freq, off_set, train_class=train)
