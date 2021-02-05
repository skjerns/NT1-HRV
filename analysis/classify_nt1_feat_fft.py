# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 using the matrix profile

@author: Simon Kern
"""
import matplotlib.pyplot as plt
import numpy as np
import misc
from tqdm import tqdm
from sleep import SleepSet
import config as cfg
import functions
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import RFECV
# from scipy.signal import convolve
from scipy.signal import resample
from scipy import fft
flatten = lambda t: [item for sublist in t for item in sublist]
np.random.seed(0)

if True:
    plt.close('all')
    ss = SleepSet(cfg.folder_unisens)
    # ss = ss.stratify() # only use matched participants
    ss = ss.filter(lambda x: x.duration < 60*60*11) # only less than 11 hours
    ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # no hypersomnia
    ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True)[:length])<0.25) # remove artefacts >25%

    p = ss[1]
    length = int(2 * 60 * 3.5) # first four hours
    factor = 3
    name = f'RFC-feat-fft-downsample{factor}'
    #%% load data
    data_x = []
    data_y = []

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
    clf = RandomForestClassifier(2000, n_jobs=4)
    cv = StratifiedKFold(n_splits = 20, shuffle=True)
    
    y_pred = cross_val_predict(clf, data_x, data_y, cv=cv, method='predict_proba', n_jobs=5, verbose=10)

    misc.save_results(data_y, y_pred, name, ss=ss, clf=clf)
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


