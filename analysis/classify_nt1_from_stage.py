# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 when a scoring is available

@author: Simon Kern
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sleep import SleepSet
import config as cfg
import functions
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
flatten = lambda t: [item for sublist in t for item in sublist]


if True:
    plt.close('all')
    ss = SleepSet(cfg.folder_unisens)
    # ss = ss.stratify() # only use matched participants
    p = ss[1]

    #%% load data
    stages = np.arange(5)
    
    x_train = []
    y_train = []
    
    for p in tqdm(ss, desc='Loading features'):
        feature_names = []
        names = []
        hypno = p.get_hypno(only_sleeptime=True)
        feats = []
        hypno = hypno[:180]

        # features from hypnogram
        feats += [functions.starting_sequence(hypno)]
        feats += functions.count_transitions(hypno)
        feature_names.extend(['starting_seq']+[f'trans_count_{trans}' for trans in range(15)])

        # feature per sleepstage / block
        for feat_name in cfg.mapping_feats:
            if feat_name not in features.__dict__: continue
            names.append(feat_name)
            data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30, offset=True, cache=False)
            data = data[:180]
            # # first per stage
            feat_mean = [np.nanmean(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
            # # feat_median = [np.nanmedian(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
            # # feat_min = [np.nanmin(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
            # # feat_max = [np.nanmax(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
            # # feat_std = [np.nanstd(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]

            # # feat_quantiles = [[np.nanpercentile(data[hypno==stage], q) if (hypno==stage).any() else np.nan for stage in stages] for q in range(10,100, 10)]
            # # feat_quantiles = flatten(feat_quantiles)
            feats +=   feat_mean# + feat_median + feat_min + feat_max + feat_std

            for stage in stages:
                feature_names.extend([f'{feat_name}_{mod}_stage{stage}' for mod in ['mean']])#, 'median', 'min', 'max', 'std']])

            # then per 45 minute blocks
            blocks = [[x*90, x*90+90] for x in range(6)]
            feat_mean = [np.nanmean(data[start:end])  for start, end in blocks]
            # feat_median = [np.nanmedian(data[start:end]) for start, end in blocks]
            # feat_min = [np.nanmin(data[start:end]) for start, end in blocks]
            # feat_max = [np.nanmax(data[start:end]) for start, end in blocks]
            # feat_std = [np.nanstd(data[start:end]) for start, end in blocks]
            # # feat_quantiles = [[np.nanpercentile(data[start:end], q)  for start, end in blocks] for q in range(10,100, 10)]
            # # feat_quantiles = flatten(feat_quantiles)

            feats +=   feat_mean# + feat_median + feat_min + feat_max + feat_std

            for block, _ in blocks:
                feature_names.extend([f'{feat_name}_{mod}_block{block}' for mod in ['mean']])#, 'median', 'min', 'max', 'std']])

        x_train.append(feats)
        y_train.append(p.group=='nt1')

    if len(feature_names) != len(feats): print(f'Names do not match length of feature vector {len(feats)}!={len(feature_names)}')
    x_train = np.array(x_train)
    # replace nan values with the mean of this value over all other participants
    for i in range(x_train.shape[-1]):
        x_train[np.isnan(x_train[:,i]), i] = np.nanmean(x_train[:,i])
    x_train = np.nan_to_num(x_train, posinf=0, neginf=0)
    y_train = np.array(y_train)

    
    #%% train
    clf = RandomForestClassifier(1000)
    results = cross_validate(clf, x_train, y_train, cv=10, scoring=
                       ['recall', 'precision', 'accuracy', 'f1'], n_jobs=16, verbose=100)
    
    print(''.join([f'{key}: {np.mean(values):.3f}\n' for key, values in results.items()]))

    #%% feat select
    selector = RFECV(clf, cv=5, n_jobs=-1, verbose=100, step=1, scoring='f1')
    selector.fit(x_train, y_train)
    selector.grid_scores_

    print(f'These were the {selector.n_features_} features that were deemed important')
    print([feature_names[i] for i in np.nonzero(selector.support_)[0]])

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


