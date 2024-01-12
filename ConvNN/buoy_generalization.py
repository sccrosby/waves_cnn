import os
import sys; sys.path.append(os.getcwd())
from data_utils.batcher import Dataset
import torch
import torch.nn as nn

history = 24
forecast = 24
q_freq = [0.04, 0.25]
offset = 1


# class to hold datasets and files
class Buoy:
    def __init__(self, buoy_num):
        self.buoy = buoy_num
        self.files = self.get_file_config(buoy_num)
        self.train = Dataset(self.files['train_file'], 'train', history, forecast, q_freq, offset)
        self.dev = Dataset(self.files['dev_file'], 'dev', history, forecast, q_freq, offset, self.train)
        self.test = Dataset(self.files['test_file'], 'test', history, forecast, q_freq, offset, self.train)
        self.model = self.files['model']

    def get_file_config(self, buoy_num):
        data_location = '/home/hutch_research/data/waves/datasets/'
        model_location = '/home/hutch_research/projects/ml_waves19_jonny/models/'

        if buoy_num == 46211:
            files = {
                'train_file': data_location + '46211_waves_TRAIN_2019-03-03.npz',
                'dev_file':   data_location + '46211_waves_DEV_2019-03-03.npz',
                'test_file':  data_location + '46211_waves_TEST_2019-05-16.npz',
                'model':      model_location + '46211_20190601113247_dev'
            }
        elif buoy_num == 46214:
            files = {
                'train_file': data_location + '46214_waves_TRAIN_2019-03-29.npz',
                'dev_file':   data_location + '46214_waves_DEV_2019-03-29.npz',
                'test_file':  data_location + '46214_waves_TEST_2019-03-29.npz',
                'model':      model_location + '46214_20190525165851_dev'
            }
        elif buoy_num == 46218:
            files = {
                'train_file': data_location + '46218_waves_TRAIN_2019-03-29.npz',
                'dev_file':   data_location + '46218_waves_DEV_2019-03-29.npz',
                'test_file':  data_location + '46218_waves_TEST_2019-03-29.npz',
                'model':      model_location +'46218_20190525201735_dev'
            }
        else:
            raise Exception("Non-existent buoy")
        return files


# load the buoys
buoys = {'46211': Buoy(46211), '46214': Buoy(46214), '46218': Buoy(46218)}


def get_model_loss(model, dataset, mean, std):
    loss_function = nn.MSELoss()
    x, y_ww3, y_buoy = dataset.get_mini_batch(mb_size=None, full=True, as_torch_cuda=True)
    with torch.no_grad():
        pred, _ = model(x, y_ww3, mean, std)
        loss = loss_function(pred, y_buoy).item()
    # x = y_ww3 = y_buoy = None  # to clear GPU memory
    return loss


# method to test the generalization of model buoy to two other buoys
def test_buoy_generalization(model_buoy, test_buoy):
    model_buoy = buoys[model_buoy]
    test_buoy = buoys[test_buoy]
    model = torch.load(model_buoy.model)

    # get the means and std's for each
    model_buoy_mean =torch.from_numpy(model_buoy.train.mean).float().cuda()
    test_buoy_mean = torch.from_numpy(test_buoy.train.mean).float().cuda()

    model_buoy_std = torch.from_numpy(model_buoy.train.std).float().cuda()
    test_buoy_std = torch.from_numpy(test_buoy.train.std).float().cuda()

    # Get losses for test buoy's own standardization scheme
    print("buoy", str(model_buoy.buoy) + "'s", 'model with data from', test_buoy.buoy)
    dev_loss = get_model_loss(model, test_buoy.dev, test_buoy_mean, test_buoy_std)
    test_loss = get_model_loss(model, test_buoy.test, test_buoy_std, test_buoy_std)
    print("\t%s standardization-->\tdev: %.5f\ttest: %.5f" % (str(test_buoy.buoy), dev_loss, test_loss))

    # Get losses for model buoy's standardisation scheme
    dev = Dataset(test_buoy.files['dev_file'], 'dev', history, forecast, q_freq, offset, model_buoy.train)
    test = Dataset(test_buoy.files['test_file'], 'dev', history, forecast, q_freq, offset, model_buoy.train)
    dev_loss = get_model_loss(model, dev, model_buoy_mean, model_buoy_std)
    test_loss = get_model_loss(model, test, model_buoy_mean, model_buoy_std)
    print("\t%s standardization-->\tdev: %.5f\ttest: %.5f\n\n" % ('model', dev_loss, test_loss))


# 46211 --> (46214, 46218)
test_buoy_generalization('46211', '46214')
test_buoy_generalization('46211', '46218')

# 46214 --> (46211, 46218)
test_buoy_generalization('46214', '46211')
test_buoy_generalization('46214', '46218')

# 46218 --> (46211, 46214)
test_buoy_generalization('46218', '46211')
test_buoy_generalization('46218', '46214')







