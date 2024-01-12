import numpy as np

# function [md1,md2,spr1,spr2,skw,kur,sdmd1,sdspr1,spr2_H]=getkuikstats(a1,b1,a2,b2,dof)
# GETKUIKSTATS returns directional spectrum distribution stats in
# each freqeuncy band as defined by
# Kuik et. al., A method for routine analysis of pitch-and-roll
# data, JPO, 18, 1020-1034, 1988.
# %
# %
#  [md1,md2,spr1,spr2,skw,kur,sdmd1,sdspr1]=getkuikstats(a1,b1,a2,b2,dof)
# Or [md1,md2,spr1,spr2,skw,kur,sdmd1,sdspr1]=getkuikstats(d,dof)
# where d = [a1 b1 a2 b2] and is 4xN;
# input: low order normalized directional fourier coefficients a1,b1,a2,b2
#
# output:
#
#  md1= 1st moment mean direction 
#  md2= 2nd moment mean direction
#  spr1= 1st moment spread
#  spr2= 2nd moment spread
#  skw = skewness
#  kur = kurtosis
#  sdm1 = standard deviation of m1
#  sdspr1 = standard deviation of spr13

# good reference:
# https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

eps = np.spacing(1)

def get_kuik_stats(a1, b1, a2, b2, dof=32):

    num_args = 0
    # TODO: cases for num_args
    if a1 is not None: num_args += 1
    if b1 is not None: num_args += 1
    if a2 is not None: num_args += 1
    if b2 is not None: num_args += 1

    # first moment mean direction (radians)
    md1r = np.arctan2(b1 + eps, a1 + eps)
    md1 = np.rad2deg(md1r)  # in degrees

    # turn negative directions in positive directions
    # MATLAB: md1(md1 < 0)=md1(md1 < 0)+360;
    md1[md1 < 0] = md1[md1 < 0] + 360

    # first moment spread
    spr = 2 * (1 - np.sqrt(a1 ** 2 + b1 ** 2))
    spr1 = np.sqrt(spr) * 180/np.pi

    # second moment mean direction in degrees
    md2 = 0.5 * np.arctan2(b2, a2) * (180 / np.pi)
    # turn negative directions in positive directions
    md2[md2 < 0] = md2[md2 < 0] + 360
    # a2b2 mean dir has 180 deg amiguity. find one that is closest to a1b1 mean dir.
    tdif = np.abs(md1 - md2)
    md2[tdif > 90] = md2[tdif > 90] - 180
    md2[md2 < 0] = md2[md2 < 0] + 360

    # second moment spread
    m2 = a2 * np.cos(2 * md1r) + b2 * np.sin(2 * md1r)
    spr2 = np.sqrt((1.0 - m2) / 2) * (180 / np.pi)

    spr2_H = 180 / np.pi * np.sqrt((1 - np.sqrt(a2 ** 2 + b2 ** 2)) / 2)

    # skewness & kurtosis

    # m1 after Kuik et. al.
    rm1 = np.sqrt(a1 ** 2 + b1 ** 2)
    # 2 times mean direction
    t2 = 2 * np.arctan2(b1, a1)
    # n2 after Kuik et. al.
    rn2 = b2 * np.cos(t2) - a2 * np.sin(t2)
    # m2 after Kuik et. al.
    rm2 = a2 * np.cos(t2) + b2 * np.sin(t2)

    # kurtosis_1
    kur = (6. - 8. * rm1 + 2. * rm2) / ((2 * (1. - rm1)) ** 2)
    # skewness_1
    skw = -rn2 / (.5 * (1 - rm2)) ** 1.5

    # Use Kuik, 1988 Eqs 40,41 to estimate standard deviation which is very
    # near to rms error as bias is an order of magnitude less
    # s.d.(md1)
    sdmd1 = dof ** (-.5) * np.sqrt((1 - rm2) / (2 * rm1 ** 2))
    sdmd1 = sdmd1 * (180 / np.pi)
    # s.d.(spr1)
    # sdspr1 = dof ** (-.5) * np.sqrt(rm1 ** 2 / (2 * (1 - rm1)) * (rm1 ** 2 + (rm2 ** 2 + rn2 ** 2 - 1) / 4 + (1 + rm2) ** (rm1 ** (-2) - 2) / 2))
    sdspr1 = rm1 ** 2
    sdspr1 = sdspr1 + (rm2 ** 2 + rn2 ** 2 - 1) / 4
    sdspr1 = sdspr1 + (1 + rm2) ** (rm1 ** (-2) - 2) / 2
    sdspr1 = rm1 ** 2 / (2 * (1 - rm1)) * (sdspr1)
    sdspr1 = dof ** (-.5) * np.sqrt(sdspr1)
    sdspr1 = sdspr1 * (180 / np.pi)

    return md1, md2, spr1, spr2, skw, kur, sdmd1, sdspr1, spr2_H
