# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 using the matrix profile

@author: Simon Kern
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sleep import SleepSet
import config as cfg
import pandas as pd
import functions
import features
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
# from scipy.signal import convolve
from scipy.ndimage.filters import convolve
from scipy import fft
flatten = lambda t: [item for sublist in t for item in sublist]


if True:
    plt.close('all')
    ss = SleepSet(cfg.folder_unisens)
    # ss = ss.stratify() # only use matched participants
    p = ss[1]

    #%% load data
    x_train = []
    y_train = []

    factor = 5

    for p in tqdm(ss, desc='Loading features'):
        feats = {}
        # feature per sleepstage / block
        for feat_name in cfg.mapping_feats:
        # for feat_name in [feat_name]:
            if feat_name not in features.__dict__: continue
            data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30, offset=True, cache=False)
            data = functions.interpolate_nans(data)
            data = data[:900]
            feat = fft.fft(data)[:50]
            # real = zscore(convolve(feat.real, np.ones(factor)/factor)[::factor])
            imag = convolve(feat.imag, np.ones(factor)/factor)[::factor]
            # feats[f'{feat_name}_real'] = real
            feats[f'{feat_name}_imag'] = imag

            if feat_name!='mean_HR': continue


        x_train.append(np.array([x for x in feats.values()]).ravel())
        y_train.append(p.group=='nt1')


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    feature_names = list(feats)

    #%% train
    clf = RandomForestClassifier(1000)
    results = cross_validate(clf, x_train, y_train, cv=10, scoring=
                       ['recall', 'precision', 'accuracy', 'f1'], n_jobs=16, verbose=100)
    
    print(''.join([f'{key}: {np.mean(values):.3f}\n' for key, values in results.items()]))

    #%% feature importance analysis
    clf.fit(x_train, y_train)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names_sorted = [feature_names[i] for i in indices]

    plot_n = 20

    plt.bar(np.arange(plot_n),importances[indices][:plot_n])
    plt.xticks(np.arange(plot_n), feature_names_sorted[:plot_n],rotation=70)
    plt.subplots_adjust(bottom=0.6)
    plt.title('First 15 most important features for NT1 classification')


