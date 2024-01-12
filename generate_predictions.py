from data_utils.batcher import Dataset
from ConvNN.cnn import ConvNN
import sys


def generate_predictions(model, buoy='46211'):
    history = 36
    forecast = 24
    q_freq = [0.04, 0.25]
    offset = 1

    # data files
    data_dir = "/home/hutch_research/data/waves/datasets/"
    train_fn = data_dir + "46211_waves_TRAIN_2019-03-03.npz"
    dev_fn   = data_dir + "46211_waves_DEV_2019-03-03.npz"
    test_fn  = data_dir + "46211_waves_TEST_2019-03-03.npz"

    # model file
    model_dir = "/home/hutch_research/projects/ml_waves19_jonny/models/"
    model_fn = model_dir + model

    # Load the dataset -- this will take a moment
    trn_data = Dataset(train_fn, 'train', history, forecast, q_freq, offset)
    dev_data = Dataset(dev_fn, 'dev', history, forecast, q_freq, offset, trn_data)

    # create the predictions
    dev_data.create_predictions(model_fn, True, None)


if __name__ == '__main__':
    assert(len(sys.argv) > 1)

    generate_predictions(sys.argv[1])
