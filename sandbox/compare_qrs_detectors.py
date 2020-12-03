# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:43:11 2020

Compare different publicly available QRS detection packages

@author: Simon
"""
import stimer
import sys, os
from pyedflib import highlevel
from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from viewer import viewer
from functions import resample
from external import processing
from ecgdetectors import Detectors

import config as cfg
import functions
import features
from sleep import SleepSet
import scipy
from joblib import Memory
from sklearn.neighbors import KDTree

memory = Memory('z:\\cache\\')

@memory.cache
def rpeak_gqrs(ecg, fs):
    rr_min = 0.25
    rr_max = 1.5
    res = processing.gqrs_detect(sig=ecg, fs=fs, RRmin=rr_min, RRmax=rr_max)
    return res

def calc_lag(r_true, r_pred):
    """
    most detection methods are for live detection, that means
    they delay the actual detected peak and put it at a constant rate

    """
    y_tree = KDTree(np.vstack([r_true, np.zeros_like(r_true)]).T, p=1)
    dist, idxs = y_tree.query(np.vstack([r_pred, np.zeros_like(r_pred)]).T,
                              k = 1, return_distance=True)

def compare(r_true, r_pred):
    """
    R positions are annotated in seconds
    """
    y_tree = KDTree(np.vstack([r_true, np.zeros_like(r_true)]).T, p=1)
    dist, idxs = y_tree.query(np.vstack([r_pred, np.zeros_like(r_pred)]).T,
                              k = 1, return_distance=True)
    return dist, idxs


ss = SleepSet(cfg.folder_unisens)
ss = ss.filter(lambda x:x.ecg.sampleRate==256)
p = ss[0]
r_true = p.get_RR(offset=False)[0]
ecg = p.ecg.get_data('ecg').squeeze()
fs = p.ecg.sampleRate



detectors = Detectors(fs)
detector_names =  ['engzee_detector', 'hamilton_detector', 'christov_detector',
                   'swt_detector', 'pan_tompkins_detector', 'two_average_detector']


res = {}

for name in detector_names:
    stimer.start(name)
    func = Detectors.__dict__[name]
    r_pred = np.array(func(detectors, ecg))/fs
    res[name] = r_pred
    stimer.stop(name)
    p.plot(markers={name: r_pred}, nrows=2, ncols=1, interval=60)


fig, axs= plt.subplots(2, 3)
axs = axs.flatten()
for i, name in enumerate(detector_names):
    r_pred = res[name]
    dist, idxs = compare(r_true, r_pred)
    axs[i].hist(dist, 100)
    axs[i].set_title(name)
    axs[i].set_xlim(0,0.4)
    axs[i].set_xlabel('diff in sec')
    axs[i].set_ylabel('count')
    plt.suptitle('Peak true/pred distances')


fig, axs= plt.subplots(2, 3)
axs = axs.flatten()
for i, name in enumerate(detector_names):
    r_pred = res[name]
    axs[i].plot(np.diff(r_pred), alpha=0.7)
    axs[i].plot(np.diff(r_true), alpha=0.7)
    axs[i].set_title(name)
    axs[i].set_xlabel('time')
    axs[i].set_ylabel('RR')
    axs[i].set_ylim(np.diff(r_true).min()-0.5, np.diff(r_true).max()+0.5)
    plt.legend([name, 'kubios'])

plot_begin = 0 # start point in seconds
plot_end = 60 # stop point in seconds

# plt.figure()
# plt.plot(ecg[fs*plot_begin:fs*plot_end], linewidth=0.4)

# all_methods = [res1, res2]
# for peaks in all_methods:
#     peaks = peaks[np.argmax(peaks>plot_begin):np.argmax(peaks>plot_end)]
#     x = peaks*fs-plot_begin*fs
#     y = ecg[(peaks*fs).astype(int)]
#     plt.scatter(x, y)
# plt.legend(['ECG', 'Kubios','XQRS'])
stop
#%%
for p in tqdm(ss):
    fs = p.ecg.sampleRate
    sig = p.ecg.get_data('ecg').squeeze()
    r_true = p.get_RR(offset=False)[0]

    detectors = Detectors(fs)
    r_pred = np.array(detectors.swt_detector(sig))/fs
    dist, idxs = compare(r_true, r_pred)

    x, c = np.unique(np.round(dist, 3), return_counts=True)
    most_freq = x[np.argmax(c)]

    print(f'swt : {int(most_freq*1000)}ms')
    plt.figure()
    plt.hist(dist, 100)
    plt.title('swt')
    plt.xlim(0,0.4)
    plt.xlabel('diff in sec')
    plt.ylabel('count')

