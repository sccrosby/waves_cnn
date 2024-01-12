import os
import glob
import sys; sys.path.append(os.getcwd())
from data_utils.forecast_batcher2 import Dataset

def get_file_config(buoy):
    fp = 'datasets/forecast_data/'
    print("Buoy: ",buoy)
    files = {
        'train_file': glob.glob(fp + str(buoy) + '_waves_TRAIN_*.npz')[-1],
        'dev_file':   glob.glob(fp + str(buoy) + '_waves_DEV_*.npz')[-1],
        'test_file':  glob.glob(fp + str(buoy) + '_waves_TEST_*.npz')[-1],
        }
    return files

def main(model, buoy, history, forecast, q_freq, offset, standardization, moment_shared_std, moment_div_e, src_buoy=None):

    files = get_file_config(buoy)

    if src_buoy is None:
        trn_data = Dataset(files['train_file'], 'train', history, forecast, q_freq, offset,
                           standardization=standardization, moment_shared_std=moment_shared_std, moment_div_e=moment_div_e)
    else:
        src_file = get_file_config(src_buoy)['train_file']
        trn_data = Dataset(data_dir + src_file, 'train', history, forecast, q_freq, offset)
    dev_data = Dataset(files['dev_file'],   'dev', history, forecast, q_freq, offset, trn_data,
                       standardization=standardization, moment_shared_std=moment_shared_std, moment_div_e=moment_div_e)
    tst_data = Dataset(files['test_file'],  'test', history, forecast, q_freq, offset, trn_data,
                       standardization=standardization, moment_shared_std=moment_shared_std, moment_div_e=moment_div_e)

    dev_data.create_predictions(buoy, model, False)
    tst_data.create_predictions(buoy, model, False, test_set=True)


if __name__ == '__main__':
    history = 6 
    forecast = 12
    q_freq = [0.04, 0.25]
    offset = 6

#    buoy = 46211  # Gray's Harbor
#    buoy = 46214  # Point Reyes
    buoy = 46218  # Harvest
#    buoy = 46015  # 
#    buoy = 46027  # 
#    buoy = 46029  # 
#    buoy = 46089  # 

    standardization = False
    moment_shared_std = False
    moment_div_e = True

    model_dir = '/research/hutchinson/projects/ml_waves19/ml_waves19_noah/models/'
#    model = 'GraysHarbor/46211_20210731174328_dev'
#    model = 'PointReyes/46214_20210731215833_dev'
#    model = 'Harvest/46218_20210801110042_dev'
    model_fn = model_dir + model

    main(model_fn, buoy, history, forecast, q_freq, offset, standardization, moment_shared_std, moment_div_e, src_buoy=None)


