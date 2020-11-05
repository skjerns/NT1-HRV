# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:43:11 2020

Compare different publicly available QRS detection packages

@author: Simon
"""
import sys, os
from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from viewer import viewer

import config as cfg
import functions
import features
from sleep import SleepSet
from joblib import Memory
memory = Memory('z:\\cache\\')

@memory.cache
def rpeak_gqrs(ecg, fs):
    rr_min = 0.25
    rr_max = 1.5
    res = processing.gqrs_detect(sig=ecg, fs=fs, RRmin=rr_min, RRmax=rr_max)
    return res

@memory.cache
def rpeak_xqrs(ecg, fs):
    rr_min = 0.25
    rr_max = 1.5
    res = processing.xqrs_detect(sig=ecg, fs=fs)
    return res

ss = SleepSet(cfg.folder_unisens)
p = ss[1]
ecg = p.get_ecg()
fs = p.ecg.sampleRate

res1 = p.get_RR()[0]
res2 = rpeak_gqrs(ecg, fs)/fs
res3 = rpeak_xqrs(ecg, fs)/fs

#%%
plot_begin = 60 # start point in seconds
plot_end = 90 # stop point in seconds

plt.figure()
plt.plot(ecg[fs*plot_begin:fs*plot_end])

all_methods = [res1, res2, res3]
for peaks in all_methods:
    peaks = peaks[np.argmax(peaks>plot_begin):np.argmax(peaks>plot_end)]
    x = peaks*fs-plot_begin*fs
    y = ecg[(peaks*fs).astype(int)]
    plt.scatter(x, y)

plt.legend(['ECG', 'Kubios', 'GQRS', 'XQRS'])

markers = {'Kubios': res1, 'xQRS': res3}

QRS = viewer.ECGPlotter(ecg, fs=fs, markers=markers, nrows=1, ncols=1)
