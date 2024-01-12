import os
import sys; sys.path.append(os.getcwd())
<<<<<<< HEAD
from ConvNN.cnn import ConvNN
from data_utils.batcher import Dataset
from data_utils.constants import get_bw_from_freq, A1, A2, B1, B2, E
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
import chocolate as choco
import math

# Sacred experiments are sacred
from sacred import Experiment
from sacred.observers import MongoObserver
BUOY = 46211  # Gray's Harbor
# BUOY = 46214  # Point Reyes
# BUOY = 46218  # Harvest
tm = {"h": 24, "f": 24}
num_epochs = 500
purpose = "_exp_"
EXPERIMENT_NAME = str(BUOY) + "_0618_0723_" + str(tm['h']) + '-' + str(tm['f']) + purpose + str(num_epochs)
standardization = False
moment_shared_std = False
moment_div_e = True

ex = Experiment(EXPERIMENT_NAME)

# Mongo db
# DATABASE_URL = '140.160.139.44:27017'  # hut-01
DATABASE_URL = '140.160.139.51:27017'  # hut-08
DATABASE_NAME = 'waves_mongo_tuning'  # old: 'waves_mongo'
ex.observers.append(MongoObserver.create(url=DATABASE_URL, db_name=DATABASE_NAME))

# chocolate setup
choco_conn = choco.MongoDBConnection(url=DATABASE_URL, database=DATABASE_NAME + '_' + EXPERIMENT_NAME)
# choco_conn.clear()  # clear the database for new experiment runs
=======
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chocolate as choco
import math
from ConvNN.cnn import ConvNN
from data_utils.batcher import Dataset
from data_utils.constants import get_bw_from_freq, A1, A2, B1, B2, E
from datetime import datetime

# BUOY = 46211  # Gray's Harbor
BUOY = 46214  # Point Reyes
# BUOY = 46218  # Harvest

# tm = {"h": 6, "f": 24}
tm = {"h": 6, "f": 12}
num_epochs = 500
purpose = "_exp_"
EXPERIMENT_NAME = str(BUOY) + "_0618_0723_" + str(tm['h']) + '-' + str(tm['f']) + purpose + str(num_epochs)
standardization = False
moment_shared_std = False
moment_div_e = True


buoy = BUOY
history = tm['h']
forecast = tm['f']
q_freq = [0.04, 0.25]
offset = 1

files = {
            'file_prefix': '/research/hutchinson/projects/ml_waves20_raleigh/datasets/',
            'train_file': '46211_waves_TRAIN_2019-03-03.npz',
            'dev_file':   '46211_waves_DEV_2019-03-03.npz',
            'test_file':  '46211_waves_TEST_2019-05-16.npz',
        }

train_fn = files['file_prefix'] + files['train_file']
dev_fn = files['file_prefix'] + files['dev_file']
test_fn = files['file_prefix'] + files['test_file']
best_model_fn = "models/" + str(BUOY) + '_' + datetime.now().strftime("%Y%m%d%H%M%S")

epochs = 500
on_gpu = True if torch.cuda.is_available else False
mb_size = 16

learning_rate = 0.00001
momentum = None
weight_decay = 0

h_filter = 3
v_filter = 3
d_filter = None  # num layers to reduce size
u_filter = (32, 32, 64, 64, 128, 128, 256, 256)

# chocolate_token = 0

relu_e = False

md_lambda_weight = 0.0
md_lambda = md_lambda_weight / (2 * math.pi)

moment_constraint = 'hardtanh'

energy_lambda = 0.0

flag_standardization = standardization
flag_moment_shared_std = moment_shared_std
flag_moment_div_e = moment_div_e
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101


#
def get_file_config():
<<<<<<< HEAD
    data_location = '/home/hutch_research/data/waves/datasets/'
=======
    data_location = '/research/hutchinson/projects/ml_waves20_raleigh/data/waves/datasets/'
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
    if BUOY == 46211:
        files = {
            'file_prefix': data_location,
            'train_file': '46211_waves_TRAIN_2019-03-03.npz',
            'dev_file':   '46211_waves_DEV_2019-03-03.npz',
            'test_file':  '46211_waves_TEST_2019-05-16.npz',
        }
    elif BUOY == 46214:
        files = {
            'file_prefix': data_location,
            'train_file': '46214_waves_TRAIN_2019-03-29.npz',
            'dev_file':   '46214_waves_DEV_2019-03-29.npz',
            'test_file':  '46214_waves_TEST_2019-03-29.npz',
        }
    elif BUOY == 46218:
        files = {
            'file_prefix': data_location,
            'train_file': '46218_waves_TRAIN_2019-03-29.npz',
            'dev_file':   '46218_waves_DEV_2019-03-29.npz',
            'test_file':  '46218_waves_TEST_2019-03-29.npz',
        }
    else:
        files = None
    return files
<<<<<<< HEAD


# Sacred configs
@ex.config
def train_configs():
    buoy = BUOY
    history = tm['h']
    forecast = tm['f']
    q_freq = [0.04, 0.25]
    offset = 1

    files = get_file_config()

    train_fn = files['file_prefix'] + files['train_file']
    dev_fn = files['file_prefix'] + files['dev_file']
    test_fn = files['file_prefix'] + files['test_file']
    best_model_fn = "models/" + str(BUOY) + '_' + datetime.now().strftime("%Y%m%d%H%M%S")

    epochs = 20
    on_gpu = True if torch.cuda.is_available else False
    mb_size = 64

    learning_rate = 0.05
    momentum = None
    weight_decay = 0

    h_filter = 3
    v_filter = 3
    d_filter = None  # num layers to reduce size
    u_filter = (8, 8, 16, 16, 32, 32)
=======
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101

    # chocolate_token = 0

    relu_e = False

    md_lambda_weight = 0.0
    md_lambda = md_lambda_weight / (2 * math.pi)

    moment_constraint = None

    energy_lambda = 0.0

    flag_standardization = standardization
    flag_moment_shared_std = moment_shared_std
    flag_moment_div_e = moment_div_e


class Trainer:
    def __init__(self):
<<<<<<< HEAD
        self.history, self.forecast, self.offset, self.q_freq = self.set_data_params()
        self.epochs, self.on_gpu, self.mb_size = self.set_hyper_params()
        self.h_filter, self.v_filter, self.d_filter, self.u_filter, self.relu_e, self.moment_constraint = self.set_model_params()
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.training_dataset, self.dev_dataset, self.test_dataset = self.get_datasets()
=======
        global history, forecast, offset, q_freq, epochs, on_gpu, mb_size
        global h_filter, v_filter, d_filter, u_filter, relu_e, moment_constraint
        global learning_rate, momentum, weight_decay, train_fn, dev_fn, test_fn
        global prediction, target
        global loss_function, predictions, targets, md_predictions, md_targets, md_lambda, ewe, ewe_lambda
        global best_model_fn, energy_lambda


        self.history, self.forecast, self.offset, self.q_freq = self.set_data_params(history, forecast, offset, q_freq)
        self.epochs, self.on_gpu, self.mb_size = self.set_hyper_params(epochs, on_gpu, mb_size)
        self.h_filter, self.v_filter, self.d_filter, self.u_filter, self.relu_e, self.moment_constraint = self.set_model_params(h_filter, v_filter, d_filter, u_filter, relu_e, moment_constraint)
        self.model = self.make_model()
        self.optimizer = self.make_optimizer(learning_rate, momentum, weight_decay)
        self.training_dataset, self.dev_dataset, self.test_dataset = self.get_datasets(train_fn, dev_fn, test_fn)
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101

        self.mean = torch.from_numpy(self.training_dataset.mean).float()
        self.std = torch.from_numpy(self.training_dataset.std).float()
        if self.on_gpu:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    # @ e.xcapture
    def set_data_params(self, history, forecast, offset, q_freq):
        return history, forecast, offset, q_freq

    # @ e.xcapture
    def set_hyper_params(self, epochs, on_gpu, mb_size):
        return epochs, on_gpu, mb_size

<<<<<<< HEAD
    @ex.capture
=======
    # @ e.xcapture
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
    def set_model_params(self, h_filter, v_filter, d_filter, u_filter, relu_e, moment_constraint):
        return h_filter, v_filter, d_filter, u_filter, relu_e, moment_constraint

    def make_model(self):
        model = ConvNN(v_filter=self.v_filter, h_filter=self.h_filter, d_filter=self.d_filter,
                       u_filter=self.u_filter, forecast=self.forecast, history=self.history,
                       offset=self.offset, relu_e=self.relu_e, moment_constraint=self.moment_constraint)
        if self.on_gpu:
            model.cuda()
        return model

<<<<<<< HEAD
    @ex.capture
=======
    # @ e.xcapture
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
    def make_optimizer(self, learning_rate, momentum, weight_decay):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

<<<<<<< HEAD
    @ex.capture
=======
    # @ e.xcapture
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
    def get_datasets(self, train_fn, dev_fn, test_fn):
        training_data = Dataset(train_fn, 'train', self.history, self.forecast, self.q_freq, self.offset,
                                moment_shared_std=moment_shared_std, standardization=standardization, moment_div_e=moment_div_e)
        dev_data = Dataset(dev_fn, 'dev', self.history, self.forecast, self.q_freq, self.offset, training_data,
                           moment_shared_std=moment_shared_std, standardization=standardization, moment_div_e=moment_div_e)
        test_data = Dataset(test_fn, 'test', self.history, self.forecast, self.q_freq, self.offset, training_data,
                            moment_shared_std=moment_shared_std, standardization=standardization, moment_div_e=moment_div_e)
        return training_data, dev_data, test_data

    def atan2(self, x):
        x_dstd = x * self.std + self.mean
        return torch.atan2(x_dstd[:, :, B1, :], x_dstd[:, :, A1, :])

    def compute_energy_weighted_error(self, prediction, target):
        # energy_pred_denorm = y_hat[:, :, E, :] * self.std[E, :] + self.mean[E, :]
        # ewe_numerator = ((y_hat[:, :, 0:4, :] - y_ww3[:, :, 0:4, :]) ** 2) * energy_pred_denorm.unsqueeze(2)
        # ewe_denominator = self.forecast * self.mean[E, :]
        # energy_weighted_error = ewe_numerator.sum() / ewe_denominator.sum()
        energy_pred_denormalized = prediction[:, :, E, :] * self.std[E, :] + self.mean[E, :]
        ewe_numerator = ((prediction[:, :, 0:4, :] - target[:, :, 0:4, :]) ** 2) * energy_pred_denormalized.unsqueeze(2)
        ewe_denominator = self.forecast * self.mean[E, :]
        energy_weighted_error = ewe_numerator.sum() / ewe_denominator.sum()
        return energy_weighted_error

    def compute_loss(self, loss_function, predictions, targets, md_predictions, md_targets, md_lambda, ewe, ewe_lambda):
        prediction_loss = loss_function(predictions, targets)
        md_loss = md_lambda * loss_function(md_predictions, md_targets)
        ewe_loss = ewe_lambda * ewe
        return prediction_loss + md_loss + ewe_loss

<<<<<<< HEAD
    @ex.capture
    def train(self, _run, best_model_fn, q_freq, md_lambda, energy_lambda):
=======
    # @ e.xcapture
    def train(self, best_model_fn, q_freq, md_lambda, energy_lambda):
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
        loss_function = nn.MSELoss()
        num_epochs = self.epochs
        mb_size = self.mb_size
        best_train_loss = float("inf")
        best_dev_loss = float("inf")
        best_dev_pred_loss = float("inf")
        bw_range = get_bw_from_freq(q_freq)

        patience = 0
        patience_thresh = 50  # number of epochs

        print("Beginning Training")
        for ep in range(1, num_epochs + 1):
            # items for tracking loss
            running_loss = 0.0
            num_batches = 0

            # get first mini batches
            inputs, y_ww3, y_buoy = self.training_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu)

            while inputs is not None:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # predict residuals
                y_hat, md_hat = self.model(inputs, y_ww3, self.mean, self.std)

                # compute the energy weighted error [MB, self.forecast, 0:4, fr]
                energy_weighted_error = self.compute_energy_weighted_error(y_hat, y_ww3)

                # get md for the buoy
                md_y_buoy = self.atan2(y_buoy)

                # compute loss
                loss = self.compute_loss(loss_function, predictions=y_hat, targets=y_buoy,
                                         md_predictions=md_hat, md_targets=md_y_buoy, md_lambda=md_lambda,
                                         ewe=energy_weighted_error, ewe_lambda=energy_lambda)
                # loss = loss_function(y_hat, y_buoy) + md_lambda * loss_function(md_hat, md_y_buoy) + energy_lambda * energy_weighted_error
                loss.backward()
                self.optimizer.step()

                # track running loss
                running_loss += loss.item()
                num_batches += 1

                # get next minibatch
                inputs, y_ww3, y_buoy = self.training_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu)

            # captures the training loss
            train_loss = (running_loss / num_batches)
            if math.isnan(train_loss): return best_dev_pred_loss  # in case we lose our gradients, stop early
<<<<<<< HEAD
            _run.log_scalar('train_loss', train_loss, ep)

            # dev loss items for logging
            dev_li_y_hat = 0.0
            dev_li_md = 0.0
            dev_li_ewe = 0.0
            dev_running_loss = 0.0
            a1_running_loss = 0.0
            b1_running_loss = 0.0
            a2_running_loss = 0.0
            b2_running_loss = 0.0
            e_running_loss = 0.0
            pred_running_loss = 0.0
            md_running_loss = 0.0
            ewe_running_loss = 0.0
            dev_24_running_loss = 0.0

            # number of batches to compute loss items
            dev_num_batches = 0

            with torch.no_grad():

                # get first batch
                dev_inputs, dev_y_ww3, dev_y_buoy = self.dev_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu)

                while dev_inputs is not None:
                    # get minibatch to mean direction vector
                    dev_md_y_buoy = self.atan2(dev_y_buoy)

                    # generate predictions
                    dev_y_hat, dev_md_hat = self.model(dev_inputs, dev_y_ww3, self.mean, self.std)

                    # compute itemized loss
                    dev_li_y_hat += loss_function(dev_y_hat, dev_y_buoy).item()
                    dev_li_md += (md_lambda * loss_function(dev_md_hat, dev_md_y_buoy).item())
                    dev_li_ewe += (energy_lambda * self.compute_energy_weighted_error(dev_y_hat, dev_y_ww3).item())

                    # compute overall loss
                    dev_running_loss += dev_li_y_hat + dev_li_md + dev_li_ewe
                    dev_num_batches += 1

                    # special loss reporting for losses f > 24
                    if self.forecast > 24:
                        dev_24_running_loss += loss_function(dev_y_hat[:, 0:24, :, :], dev_y_buoy[:, 0:24, :, :]).item()

                    # other loss components for logging
                    a1_running_loss += loss_function(dev_y_hat[:, :, A1, :], dev_y_buoy[:, :, A1, :]).item()
                    a2_running_loss += loss_function(dev_y_hat[:, :, A2, :], dev_y_buoy[:, :, A2, :]).item()
                    b1_running_loss += loss_function(dev_y_hat[:, :, B1, :], dev_y_buoy[:, :, B1, :]).item()
                    b2_running_loss += loss_function(dev_y_hat[:, :, B2, :], dev_y_buoy[:, :, B2, :]).item()
                    e_running_loss += loss_function(dev_y_hat[:, :, E, :], dev_y_buoy[:, :, E, :]).item()
                    pred_running_loss += loss_function(dev_y_hat, dev_y_buoy).item()

                    # mean wave direction and wave height (hs) loss
                    # Note, this is computed in Radians, for reporting, attempt to bring this value between 0 and 1
                    # dev_y_hat_dstd = dev_y_hat * self.std + self.mean
                    # dev_y_buoy_dstd = dev_y_buoy * self.std + self.mean
                    md_running_loss += loss_function(dev_md_y_buoy, dev_md_hat).item()

                    # compute energy weighted loss
                    ewe_running_loss += self.compute_energy_weighted_error(dev_y_hat, dev_y_ww3).item()

                    # get next batch
                    dev_inputs, dev_y_ww3, dev_y_buoy = self.dev_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu)

=======
            # _run.log_scalar('train_loss', train_loss, ep)

            # dev loss items for logging
            dev_li_y_hat = 0.0
            dev_li_md = 0.0
            dev_li_ewe = 0.0
            dev_running_loss = 0.0
            a1_running_loss = 0.0
            b1_running_loss = 0.0
            a2_running_loss = 0.0
            b2_running_loss = 0.0
            e_running_loss = 0.0
            pred_running_loss = 0.0
            md_running_loss = 0.0
            ewe_running_loss = 0.0
            dev_24_running_loss = 0.0

            # number of batches to compute loss items
            dev_num_batches = 0

            with torch.no_grad():

                # get first batch
                dev_inputs, dev_y_ww3, dev_y_buoy = self.dev_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu)

                while dev_inputs is not None:
                    # get minibatch to mean direction vector
                    dev_md_y_buoy = self.atan2(dev_y_buoy)

                    # generate predictions
                    dev_y_hat, dev_md_hat = self.model(dev_inputs, dev_y_ww3, self.mean, self.std)

                    # compute itemized loss
                    dev_li_y_hat += loss_function(dev_y_hat, dev_y_buoy).item()
                    dev_li_md += (md_lambda * loss_function(dev_md_hat, dev_md_y_buoy).item())
                    dev_li_ewe += (energy_lambda * self.compute_energy_weighted_error(dev_y_hat, dev_y_ww3).item())

                    # compute overall loss
                    dev_running_loss += dev_li_y_hat + dev_li_md + dev_li_ewe
                    dev_num_batches += 1

                    # special loss reporting for losses f > 24
                    if self.forecast > 24:
                        dev_24_running_loss += loss_function(dev_y_hat[:, 0:24, :, :], dev_y_buoy[:, 0:24, :, :]).item()

                    # other loss components for logging
                    a1_running_loss += loss_function(dev_y_hat[:, :, A1, :], dev_y_buoy[:, :, A1, :]).item()
                    a2_running_loss += loss_function(dev_y_hat[:, :, A2, :], dev_y_buoy[:, :, A2, :]).item()
                    b1_running_loss += loss_function(dev_y_hat[:, :, B1, :], dev_y_buoy[:, :, B1, :]).item()
                    b2_running_loss += loss_function(dev_y_hat[:, :, B2, :], dev_y_buoy[:, :, B2, :]).item()
                    e_running_loss += loss_function(dev_y_hat[:, :, E, :], dev_y_buoy[:, :, E, :]).item()
                    pred_running_loss += loss_function(dev_y_hat, dev_y_buoy).item()

                    # mean wave direction and wave height (hs) loss
                    # Note, this is computed in Radians, for reporting, attempt to bring this value between 0 and 1
                    # dev_y_hat_dstd = dev_y_hat * self.std + self.mean
                    # dev_y_buoy_dstd = dev_y_buoy * self.std + self.mean
                    md_running_loss += loss_function(dev_md_y_buoy, dev_md_hat).item()

                    # compute energy weighted loss
                    ewe_running_loss += self.compute_energy_weighted_error(dev_y_hat, dev_y_ww3).item()

                    # get next batch
                    dev_inputs, dev_y_ww3, dev_y_buoy = self.dev_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu)

>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
                # primary loss components
                # dev_loss = dev_running_loss / dev_num_batches
                dev_loss = (dev_li_md + dev_li_y_hat + dev_li_ewe) / dev_num_batches
                dev_li_md = dev_li_md / dev_num_batches
                dev_li_ewe = dev_li_ewe / dev_num_batches
                dev_li_y_hat = dev_li_y_hat / dev_num_batches
<<<<<<< HEAD
                _run.log_scalar('dev_loss', dev_loss, ep)
                _run.log_scalar('dev_li_y_hat', dev_li_y_hat, ep)
                _run.log_scalar('dev_li_md', dev_li_md, ep)
                _run.log_scalar('dev_li_ewe', dev_li_ewe, ep)

                # other loss components for analysis
                _run.log_scalar('a1_loss', a1_running_loss / dev_num_batches, ep)
                _run.log_scalar('b1_loss', b1_running_loss / dev_num_batches, ep)
                _run.log_scalar('a2_loss', a2_running_loss / dev_num_batches, ep)
                _run.log_scalar('b2_loss', b2_running_loss / dev_num_batches, ep)
                _run.log_scalar('e_loss', e_running_loss / dev_num_batches, ep)
                _run.log_scalar('pred_loss', pred_running_loss / dev_num_batches, ep)
                _run.log_scalar('md_loss', md_running_loss / dev_num_batches, ep)
                _run.log_scalar('ewe_loss', ewe_running_loss / dev_num_batches, ep)
                if self.forecast > 24:
                    _run.log_scalar('dev_24_loss', dev_24_running_loss / dev_num_batches, ep)
=======
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101

            # check for loss updates
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_dev_pred_loss = pred_running_loss / dev_num_batches
                best_dev_md_loss = md_running_loss / dev_num_batches
                best_dev_epoch = ep
<<<<<<< HEAD
                ex.log_scalar('best_dev_loss', best_dev_loss, ep)
                ex.log_scalar('best_dev_pred_loss', best_dev_pred_loss, ep)
                ex.log_scalar('best_dev_md_loss', best_dev_md_loss, ep)
                ex.log_scalar('best_dev_epoch', best_dev_epoch)
                ex.log_scalar('test_loss_dev', self.check_test_loss(loss_function, md_lambda), ep)
                torch.save(self.model, best_model_fn + "_dev")
=======
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
                patience = 0
            else:
                patience += 1

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_epoch = ep
<<<<<<< HEAD
                ex.log_scalar('best_train_loss', best_train_loss, ep)
                ex.log_scalar('best_train_epoch', best_train_epoch)
                ex.log_scalar('test_loss_train', self.check_test_loss(loss_function, md_lambda), ep)
                torch.save(self.model, best_model_fn + "_train")
=======
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101

            print("epoch: %s" % str(ep), 'train MSE: %.4f' % train_loss, 'dev MSE: %.4f' % dev_loss,
                  'pred MSE: %.4f' % (pred_running_loss / dev_num_batches),
                  'md MSE: %.4f' % (md_running_loss / dev_num_batches))

            # early stopping
            if patience >= patience_thresh:
                break
        # end training loop

        # used to catch bad losses
        # if best_dev_pred_loss > 100:
        #     best_dev_pred_loss = 100.0

        # return best_dev_pred_loss + best_dev_md_loss
        return best_dev_pred_loss

    def run(self):
<<<<<<< HEAD
        best_loss = self.train()
=======
        best_loss = self.train(best_model_fn, q_freq, md_lambda, energy_lambda)
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
        return best_loss

    def check_test_loss(self, loss_function, md_lambda):
        test_running_loss = 0.0
        test_num_batches = 0
<<<<<<< HEAD

        test_inputs, test_y_ww3, test_y_buoy = self.test_dataset.get_mini_batch(self.mb_size, as_torch_cuda=self.on_gpu)
        while test_inputs is not None:

            with torch.no_grad():
                test_y_hat, test_md_hat = self.model(test_inputs, test_y_ww3, self.mean, self.std)
                test_md_y_buoy = self.atan2(test_y_buoy)

                # compute loss
                test_running_loss += loss_function(test_y_hat, test_y_buoy).item() + md_lambda * loss_function(test_md_hat, test_md_y_buoy).item()

                # test_running_loss += loss_function(test_y_hat, test_y_buoy).item()
                test_num_batches += 1

            test_inputs, test_y_ww3, test_y_buoy = self.test_dataset.get_mini_batch(self.mb_size, as_torch_cuda=self.on_gpu)

        return test_running_loss / test_num_batches
=======
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101

        test_inputs, test_y_ww3, test_y_buoy = self.test_dataset.get_mini_batch(self.mb_size, as_torch_cuda=self.on_gpu)
        while test_inputs is not None:

            with torch.no_grad():
                test_y_hat, test_md_hat = self.model(test_inputs, test_y_ww3, self.mean, self.std)
                test_md_y_buoy = self.atan2(test_y_buoy)

                # compute loss
                test_running_loss += loss_function(test_y_hat, test_y_buoy).item() + md_lambda * loss_function(test_md_hat, test_md_y_buoy).item()

                # test_running_loss += loss_function(test_y_hat, test_y_buoy).item()
                test_num_batches += 1

            test_inputs, test_y_ww3, test_y_buoy = self.test_dataset.get_mini_batch(self.mb_size, as_torch_cuda=self.on_gpu)

        return test_running_loss / test_num_batches


# @ e.xmain
def start_train(_run):
    print("here")
    trainer = Trainer()
    dev_loss = trainer.run()
    return dev_loss
<<<<<<< HEAD


def create_space():
    in_channels = 2 * tm['h'] + tm['f']

    d_ch_half  = int(0.5 * in_channels)
    d_ch_1thrd = int(1. / 3. * in_channels)
    d_ch_2thrd = int(2. / 3. * in_channels)

    space = {
        "learning_rate": choco.log(low=-7, high=-5, base=10),
        # "momentum": choco.quantized_uniform(low=0.5, high=0.91, step=0.01),
        # "history": choco.choice([3, 6, 12, 24, 48]),
        "h_filter": 3,
        "v_filter": 3,
        "mb_size": choco.choice([16, 32, 48, 64]),
        # "md_lambda_weight": choco.quantized_uniform(low=0.0, high=8.5, step=1.0),
        # "energy_lambda": choco.uniform(low=0.0001, high=0.1),
        "moment_constraint": choco.choice(['tanh', 'hardtanh']),
        "d_filter": choco.choice([None,
                                  (d_ch_half, ),
                                  (d_ch_half, d_ch_half),
                                  (d_ch_2thrd, d_ch_1thrd),
                                  (d_ch_2thrd, d_ch_2thrd, d_ch_1thrd, d_ch_1thrd),
                                  ]),
        "u_filter": choco.choice([# (8, 8),
                                  # (8, 8, 16, 16),
                                  (8, 8, 16, 16, 32, 32),
                                  (8, 8, 16, 16, 32, 32, 64, 64),
                                  (8, 8, 16, 16, 32, 32, 64, 64, 128, 128),
                                  (8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256),
                                  # (16, 16),
                                  # (16, 16, 32, 32),
                                  (16, 16, 32, 32, 64, 64),
                                  (16, 16, 32, 32, 64, 64, 128, 128),
                                  (16, 16, 32, 32, 64, 64, 128, 128, 256, 256),
                                  # (32, 32),
                                  # (32, 32, 64, 64),
                                  (32, 32, 64, 64, 128, 128),
                                  (32, 32, 64, 64, 128, 128, 256, 256),
                                  ]),
    }

    return space
=======
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101


if __name__ == '__main__':
    num_experiments = 3
<<<<<<< HEAD
    choco_space = create_space()
    sampler = choco.Bayes(choco_conn, choco_space, clear_db=False)

    for i in range(num_experiments):
        token, params = sampler.next()

        # tm = {"h": 3, "f": 24}
        # tm['h'] = params['history']

        # Add additional params
        # params['chocolate_token'] = token['_chocolate_id']
        params['epochs'] = num_epochs

        loss = ex.run(config_updates=params).result

        sampler.update(token, loss)
=======

    for i in range(num_experiments):
        params = {
            #'gpu': 'y',
            'epochs': 500,
            'mb_size': 16,
            'v_filter': 1,
            'h_filter': 3,
            'd_filter': None,
            'u_filter': '32 32 64 64 128 128 256 256',
            'forecast': 12,
            'history': 6,
            'offset': 6,
        }

        start_train(params)
>>>>>>> 09b78a88584108398a11bc0ab54af62af03f3101
