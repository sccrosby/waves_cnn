import os
import sys; sys.path.append(os.getcwd())
from data_utils.batcher import Dataset

def get_file_config(buoy):
    data_location = '/research/hutchinson/projects/ml_waves19/ml_waves19_noah/datasets/'
    if buoy == 46211:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': '46211_waves_TRAIN_2019-03-03.npz',
            'dev_file':   '46211_waves_DEV_2019-03-03.npz',
            'test_file':  '46211_waves_TEST_2019-05-16.npz',
        }
    elif buoy == 46214:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': '46214_waves_TRAIN_2019-03-29.npz',
            'dev_file':   '46214_waves_DEV_2019-03-29.npz',
            'test_file':  '46214_waves_TEST_2019-03-29.npz',
        }
    elif buoy == 46218:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': '46218_waves_TRAIN_2019-03-29.npz',
            'dev_file':   '46218_waves_DEV_2019-03-29.npz',
            'test_file':  '46218_waves_TEST_2019-03-29.npz',
        }
    elif buoy == 46015:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': 'new_buoys/46015_waves_TRAIN_2021-05-27.npz',
            'dev_file':   'new_buoys/46015_waves_DEV_2021-05-27.npz',
            'test_file':  'new_buoys/46015_waves_TEST_2021-05-27.npz',
        }
    elif buoy == 46027:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': 'new_buoys/46027_waves_TRAIN_2021-05-26.npz',
            'dev_file':   'new_buoys/46027_waves_DEV_2021-05-26.npz',
            'test_file':  'new_buoys/46027_waves_TEST_2021-05-26.npz',
        }
    elif buoy == 46029:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': 'new_buoys/46029_waves_TRAIN_2021-05-27.npz',
            'dev_file':   'new_buoys/46029_waves_DEV_2021-05-27.npz',
            'test_file':  'new_buoys/46029_waves_TEST_2021-05-27.npz',
        }
    elif buoy == 46089:
        print("Buoy: ",buoy)
        files = {
            'file_prefix': data_location,
            'train_file': 'new_buoys/46089_waves_TRAIN_2021-05-27.npz',
            'dev_file':   'new_buoys/46089_waves_DEV_2021-05-27.npz',
            'test_file':  'new_buoys/46089_waves_TEST_2021-05-27.npz',
        }
    else:
        files = None
    return files

def main(model, buoy, history, forecast, q_freq, offset, standardization, moment_shared_std, moment_div_e, src_buoy=None):

    files = get_file_config(buoy)

    data_dir = files['file_prefix']

    if src_buoy is None:
        trn_data = Dataset(data_dir + files['train_file'], 'train', history, forecast, q_freq, offset,
                           standardization=standardization, moment_shared_std=moment_shared_std, moment_div_e=moment_div_e)
    else:
        src_file = get_file_config(src_buoy)['train_file']
        trn_data = Dataset(data_dir + src_file, 'train', history, forecast, q_freq, offset)
    dev_data = Dataset(data_dir + files['dev_file'],   'dev', history, forecast, q_freq, offset, trn_data,
                       standardization=standardization, moment_shared_std=moment_shared_std, moment_div_e=moment_div_e)
    tst_data = Dataset(data_dir + files['test_file'],  'test', history, forecast, q_freq, offset, trn_data,
                       standardization=standardization, moment_shared_std=moment_shared_std, moment_div_e=moment_div_e)

    dev_data.create_predictions(buoy, model, False)
    tst_data.create_predictions(buoy, model, False, test_set=True)


if __name__ == '__main__':
    history = 12
    forecast = 24
    q_freq = [0.04, 0.25]
    offset = 6

#    buoy = 46211  # Gray's Harbor
#    buoy = 46214  # Point Reyes
#    buoy = 46218  # Harvest
#    buoy = 46015  # Port Orford 
#    buoy = 46027  # Crescent City
#    buoy = 46029  # Columbia River Bar
    buoy = 46089  # Tillamook

    standardization = False
    moment_shared_std = False
    moment_div_e = True

    model_dir = '/research/hutchinson/projects/ml_waves19/ml_waves19_noah/models/'
    model = 'B46089/46089_20220113170717_dev'
#    model = 'PointReyes/46214_20210731215833_dev'
#    model = 'Harvest/46218_20210801110042_dev'
#    model = 'B46015/46015_20210801172745_dev'
    model_fn = model_dir + model

    main(model_fn, buoy, history, forecast, q_freq, offset, standardization, moment_shared_std, moment_div_e, src_buoy=None)

