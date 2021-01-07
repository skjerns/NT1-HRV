# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 using the matrix profile

@author: Simon Kern
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from misc import save_results
from tqdm import tqdm
from sleep import SleepSet
import config as cfg
import pandas as pd
import functions
import features
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
# from scipy.signal import convolve
from scipy.signal import resample
from scipy.ndimage.filters import convolve
from scipy import fft
flatten = lambda t: [item for sublist in t for item in sublist]
np.random.seed(0)

if True:
    plt.close('all')
    ss = SleepSet(cfg.folder_unisens)
    # ss = ss.stratify() # only use matched participants
    ss = ss.filter(lambda x: x.duration < 60*60*11) # only less than 11 hours
    ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # no hypersomnia
    ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True))<0.25) # remove artefacts >25%

    p = ss[1]
    length = 450

    #%% load data
    data_x = []
    data_y = []

    factor = 3

    for p in tqdm(ss, desc='Loading features'):
        feats = {}

        # feature per sleepstage / block
        for feat_name in cfg.mapping_feats:
        # for feat_name in [feat_name]:
            if feat_name not in features.__dict__: continue
            data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30)
            data = functions.interpolate_nans(data)
            data = data[:length]
            feat = 20*np.log10(np.abs(fft.fft(data)[:50])) # freq in db
            feat = resample(feat, len(feat)//factor, window='hamming')
            if np.all(np.isnan(feat)):
                feat = np.zeros_like(feat)
            feats[feat_name] = feat


        data_x.append(np.array([x for x in feats.values()]).ravel())
        data_y.append(p.group=='nt1')


    data_x = np.array(data_x)
    data_y = np.array(data_y)
    feature_names = list(feats)

    #%% train
    clf = RandomForestClassifier(10)
    cv = StratifiedKFold(shuffle=True)
    y_pred = []
    y_true = []
    for idx_train, idx_test in cv.split(data_x, data_y, groups=data_y):
        train_x = data_x[idx_train]
        train_y = data_y[idx_train]
        test_x = data_x[idx_test]
        test_y = data_y[idx_test]
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        y_pred.extend(pred)
        y_true.extend(test_y)

    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))  # once more for printing
    name = 'RFC-feat-fft'
    save_results(report, name, ss=ss, clf=clf)
    stop
    #%% feature importance analysis
    clf.fit(data_x, data_y)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names_sorted = [feature_names[i] for i in indices]

    plot_n = 20

    plt.bar(np.arange(plot_n),importances[indices][:plot_n])
    plt.xticks(np.arange(plot_n), feature_names_sorted[:plot_n],rotation=70)
    plt.subplots_adjust(bottom=0.6)
    plt.title('First 15 most important features for NT1 classification')


