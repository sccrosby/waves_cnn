import os
import sys; sys.path.append(os.getcwd())
from ConvNN.cnn import ConvNN
from data_utils.batcher import Dataset
from data_utils.constants import get_bw_from_freq
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import numpy as np
import chocolate as choco

# Sacred experiments are sacred
from sacred import Experiment
from sacred.observers import MongoObserver
# BUOY = 46211  # Gray's Harbor
# BUOY = 46214  # Point Reyes
BUOY = 46218  # Harvest
EXPERIMENT_NAME = str(BUOY)
ex = Experiment(EXPERIMENT_NAME)

# Mongo db
DATABASE_URL = '140.160.139.44:27017'
DATABASE_NAME = 'waves_mongo'
ex.observers.append(MongoObserver.create(url=DATABASE_URL, db_name=DATABASE_NAME))

# chocolate setup

choco_conn = choco.MongoDBConnection(url=DATABASE_URL, database=DATABASE_NAME + '_' + str(BUOY))
choco_conn.clear()  # clear the database for new experiment runs


#
def get_file_config():
    data_location = '/home/hutch_research/data/waves/datasets/'
    if BUOY == 46211:
        files = {
            'file_prefix': data_location,
            'train_file': '46211_waves_TRAIN_2019-03-03.npz',
            'dev_file':   '46211_waves_DEV_2019-03-03.npz',
            'test_file':  '46211_waves_TEST_2019-03-03.npz',
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


# Sacred configs
@ex.config
def train_configs():
    buoy = BUOY
    history = 36
    forecast = 24
    q_freq = [0.04, 0.25]
    offset = 1

    files = get_file_config()

    train_fn = files['file_prefix'] + files['train_file']
    dev_fn = files['file_prefix'] + files['dev_file']
    best_model_fn = "models/" + str(BUOY) + '_' + datetime.now().strftime("%Y%m%d%H%M%S_") + EXPERIMENT_NAME

    epochs = 20
    on_gpu = True if torch.cuda.is_available else False
    mb_size = 64

    learning_rate = 0.05
    momentum = 0.9
    weight_decay = 0

    h_filter = 3
    v_filter = 5
    d_filter = None
    u_filter = (8, 8, 16, 16, 32, 32)

    chocolate_token = 0

    relu_e = True


class Trainer:
    def __init__(self):
        self.history, self.forecast, self.offset, self.q_freq = self.set_data_params()
        self.epochs, self.on_gpu, self.mb_size = self.set_hyper_params()
        self.h_filter, self.v_filter, self.d_filter, self.u_filter, self.relu_e = self.set_model_params()
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.training_dataset, self.dev_dataset = self.get_datasets()

    @ex.capture
    def set_data_params(self, history, forecast, offset, q_freq):
        return history, forecast, offset, q_freq

    @ex.capture
    def set_hyper_params(self, epochs, on_gpu, mb_size):
        return epochs, on_gpu, mb_size

    @ex.capture
    def set_model_params(self, h_filter, v_filter, d_filter, u_filter, relu_e):
        return h_filter, v_filter, d_filter, u_filter, relu_e

    def make_model(self):
        model = ConvNN(v_filter=self.v_filter, h_filter=self.h_filter, d_filter=self.d_filter,
                       u_filter=self.u_filter, forecast=self.forecast, history=self.history,
                       offset=self.offset, relu_e=self.relu_e)
        if self.on_gpu:
            model.cuda()
        return model

    @ex.capture
    def make_optimizer(self, learning_rate, momentum, weight_decay):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        return optimizer

    @ex.capture
    def get_datasets(self, train_fn, dev_fn):
        training_data = Dataset(train_fn, 'train', self.history, self.forecast, self.q_freq, self.offset)
        dev_data = Dataset(dev_fn, 'dev', self.history, self.forecast, self.q_freq, self.offset, training_data)
        return training_data, dev_data

    @ex.capture
    def train(self, _run, best_model_fn, q_freq):
        loss_function = nn.MSELoss()
        num_epochs = self.epochs
        mb_size = self.mb_size
        best_dev_epoch = -1
        best_train_epoch = -1
        best_train_loss = float("inf")
        best_dev_loss = float("inf")
        bw_range = get_bw_from_freq(q_freq)

        print("Beginning Training")
        for ep in range(1, num_epochs + 1):
            # items for tracking loss
            running_loss = 0.0
            num_batches = 0

            # get first mini batches
            inputs, true_ww3, true_buoy = self.training_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu, resids=True)

            while inputs is not None:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward, loss, backward, optimize
                pred_residual = self.model(inputs)
                pred = pred_residual + true_ww3
                pred = enforce_constraints(pred)
                loss = loss_function(pred,true_buoy)
                loss.backward()
                self.optimizer.step()

                # track running loss
                running_loss += loss.item()
                num_batches += 1

                # get next minibatch
                inputs, residuals = self.training_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu, resids=True)

            # captures the training loss
            train_loss = (running_loss / num_batches)
            _run.log_scalar('train_loss', train_loss, ep)

            # capture the dev loss
            with torch.no_grad():
                dev_inp, dev_residuals = self.dev_dataset.get_mini_batch(mb_size, as_torch_cuda=self.on_gpu, full=True, resids=True)
                dev_outputs = self.model(dev_inp)
                dev_loss = loss_function(dev_outputs, dev_residuals).item()
                _run.log_scalar('dev_loss', dev_loss, ep)
                # torch.Size([16860, 24, 5, 28])
                for hour in range(dev_outputs.size()[1]):
                    hour_loss = loss_function(dev_outputs[:, hour, :, :], dev_residuals[:, hour, :, :]).item()
                    log_item = 'dev_loss_hr_' + str(hour + 1).zfill(3)
                    _run.log_scalar(log_item, hour_loss, ep)

            # check for loss updates
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_dev_epoch = ep
                ex.log_scalar('best_dev_loss', best_dev_loss, ep)
                ex.log_scalar('best_dev_epoch', best_dev_epoch)
                torch.save(self.model, best_model_fn + "_dev")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_epoch = ep
                ex.log_scalar('best_train_loss', best_train_loss, ep)
                ex.log_scalar('best_train_epoch', best_train_epoch)
                torch.save(self.model, best_model_fn + "_train")

            print("epoch: %s" % str(ep), 'train MSE: %.4f' % train_loss, 'dev MSE: %.4f' % dev_loss)
        # end for loop

        # used to catch bad losses
        if best_dev_loss > 100:
            best_dev_loss = 100.0

        return best_dev_loss

    def run(self):
        best_loss = self.train()
        return best_loss


@ex.main
def main(_run):
    trainer = Trainer()
    dev_loss = trainer.run()
    return dev_loss


def create_space():
    space = {
        "learning_rate": choco.log(low=-3, high=-1, base=10),
        "momentum": choco.quantized_uniform(low=0.5, high=0.91, step=0.01),
        "h_filter": choco.quantized_uniform(low=1, high=11, step=2),
        "v_filter": choco.quantized_uniform(low=1, high=7, step=2),
        "mb_size": choco.choice([16, 32, 48, 64]),
        "d_filter": choco.choice([None, (48,), (64, 32)]),
        "u_filter": choco.choice([(8, 8),
                                  (8, 8, 16, 16),
                                  (8, 8, 16, 16, 32, 32),
                                  (8, 8, 16, 16, 32, 32, 64, 64),
                                  (8, 8, 16, 16, 32, 32, 64, 64, 128, 128),
                                  (8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256),
                                  (16, 16),
                                  (16, 16, 32, 32),
                                  (16, 16, 32, 32, 64, 64),
                                  (16, 16, 32, 32, 64, 64, 128, 128),
                                  (16, 16, 32, 32, 64, 64, 128, 128, 256, 256),
                                  ]),
    }

    return space


if __name__ == '__main__':
    num_experiments = 100
    num_epochs = 100
    choco_space = create_space()
    sampler = choco.Bayes(choco_conn, choco_space, clear_db=True)

    for i in range(num_experiments):
        token, params = sampler.next()

        # Add additional params
        params['chocolate_token'] = token['_chocolate_id']
        params['epochs'] = num_epochs

        loss = ex.run(config_updates=params).result

        sampler.update(token, loss)
