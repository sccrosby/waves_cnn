import os
import numpy as np
import matplotlib.pyplot as plt
from data_utils.get_kuik_stats import get_kuik_stats
from data_utils.load_data import load_data_as_dict
from data_utils.matlab_datenums import matlab_datenum_to_py_date, matlab_datenum_nparray_to_py_date


if 'hutch_research' in os.getcwd():
    file_prefix = '/home/hutch_research/data/waves/'
else:
    file_prefix = '/home/jonny/PycharmProjects/ml_waves19/'


def subplot_demo():
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)

    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)

    plt.subplot(3, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('A tale of 2 subplots')
    plt.ylabel('Damped oscillation')

    plt.subplot(3, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')

    plt.subplot(3, 1, 3)
    plt.plot(x2, y2, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')

    plt.show()


def replica_plot_mean_spread(obs, ww3, ff, hr=None, hr_label="hr", path=None):
    """
    Compare mean direction and spread (derivative of fourier moments)
    """
    lw = "0.1"
    a = 0.7
    plot_hr = False if hr is None else True
    title = 'Mean direction and spread [ff=%0.3f]' % obs.fr[0, ff]
    if plot_hr:
        title += '(includes %s)' % hr_label
    
    plt.subplot(4, 1, 1)
    plt.title(title)
    if plot_hr:
        plt.plot(hr.py_date, hr.md1[:, ff], linewidth=lw, label=hr_label, alpha=a)
    plt.plot(obs.py_date, obs.md1[:, ff], linewidth=lw, label='obs', alpha=a)
    plt.plot(ww3.py_date, ww3.md1[:, ff], linewidth=lw, label='ww3', alpha=a)
    plt.ylabel('dir1')
    plt.ylim(200, 300)
    plt.yticks(np.arange(200, 301, step=20))
    plt.legend()

    plt.subplot(4, 1, 2)
    if plot_hr:
        plt.plot(hr.py_date, hr.md2[:, ff], linewidth=lw, label=hr_label, alpha=a)
    plt.plot(obs.py_date, obs.md2[:, ff], linewidth=lw, label='obs', alpha=a)
    plt.plot(ww3.py_date, ww3.md2[:, ff], linewidth=lw, label='ww3', alpha=a)
    plt.ylabel('dir2')
    plt.ylim(200, 300)
    plt.yticks(np.arange(200, 301, step=20))
    plt.legend()

    plt.subplot(4, 1, 3)
    if plot_hr:
        plt.plot(hr.py_date, hr.spr1[:, ff], linewidth=lw, label=hr_label, alpha=a)
    plt.plot(obs.py_date, obs.spr1[:, ff], linewidth=lw, label='obs', alpha=a)
    plt.plot(ww3.py_date, ww3.spr1[:, ff], linewidth=lw, label='ww3', alpha=a)
    plt.ylabel('spr1')
    plt.ylim(0, 100)
    plt.yticks(np.arange(101, step=20))
    plt.legend()

    plt.subplot(4, 1, 4)
    if plot_hr:
        plt.plot(hr.py_date, hr.spr2[:, ff], linewidth=lw, label=hr_label, alpha=a)
    plt.plot(obs.py_date, obs.spr2[:, ff], linewidth=lw, label='obs', alpha=a)
    plt.plot(ww3.py_date, ww3.spr2[:, ff], linewidth=lw, label='ww3', alpha=a)
    plt.ylabel('spr2')
    plt.ylim(0, 60)
    plt.yticks(np.arange(61, step=10))
    plt.legend()

    if path is not None:
        plt.savefig(path, dpi=500)

    plt.show()


def fetch_kuik_stats(a1, b1, a2, b2, e):
    # [O.md1, O.md2, O.spr1, O.spr2, O.skw, O.kur] = getkuikstats(O.a1. / O.e, O.b1. / O.e, O.a2. / O.e, O.b2. / O.e);
    md1, md2, spr1, spr2, skw, kur, sdmd1, sdspr1, spr2_H = get_kuik_stats(a1 / e, b1 / e, a2 / e, b2 / e)
    return md1, md2, spr1, spr2, skw, kur, sdmd1, sdspr1, spr2_H


class DataFromMat:
    def __init__(self, filename):
        data = load_data_as_dict(filename)
        self.time = data['time'].T.flatten()
        self.py_date = matlab_datenum_nparray_to_py_date(self.time.tolist())
        self.a1 = data['a1']
        self.b1 = data['b1']
        self.a2 = data['a2']
        self.b2 = data['b2']
        self.e = data['e']

        self.bw = data['bw']
        self.fr = data['fr']
        self.md1, self.md2, self.spr1, self.spr2, self.skw, self.kur, self.sdmd1, self.sdspr1, self.spr2_H = \
            fetch_kuik_stats(self.a1, self.b1, self.a2, self.b2, self.e)


if __name__ == '__main__':
    targ_dir = file_prefix + 'data_outputs/'

    obs = DataFromMat(targ_dir + '2019_03_06_200452_obs_data.mat')
    ww3 = DataFromMat(targ_dir + '2019_03_06_200452_ww3_data.mat')
    hr1 = DataFromMat(targ_dir + '2019_03_06_200452_hr1_data.mat')
    hr12 = DataFromMat(targ_dir + '2019_03_06_200452_hr12_data.mat')

    ff = 11
    replica_plot_mean_spread(obs, ww3, ff, hr=hr1, hr_label="hr1", path='hr1.png')
    replica_plot_mean_spread(obs, ww3, ff, hr=hr12, hr_label="hr12", path='hr12.png')

    # subplot_demo()
