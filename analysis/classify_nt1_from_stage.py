# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 when a scoring is available

@author: Simon Kern
"""
import sys
import os
import misc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sleep import SleepSet
import config as cfg
import functions
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
np.random.seed(0)

def flatten(t): return [item for sublist in t for item in sublist]


if True:
    plt.close('all')
    ss = SleepSet(cfg.folder_unisens)
    # ss = ss.stratify() # only use matched participants
    ss = ss.filter(lambda x: x.duration < 60*60*11)  # only less than 14 hours
    # only less than 14 hours
    ss = ss.filter(lambda x: x.group in ['control', 'nt1'])
    ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True))<0.25)
    p = ss[1]
    length = 450

    # %% load data
    stages = np.arange(5)

    data_x = []
    data_y = []

    for p in tqdm(ss, desc='Loading features'):
        feature_names = []
        names = []
        hypno = p.get_hypno(only_sleeptime=True)
        feats = []
        hypno = hypno[:length]

        # features from hypnogram
        # feats += [functions.starting_sequence(hypno)]
        # feats += functions.count_transitions(hypno)
        # feature_names.extend(['starting_seq']+[f'trans_count_{trans}' for trans in range(15)])

        # feature per sleepstage / block
        for feat_name in cfg.mapping_feats:
            if feat_name not in features.__dict__:
                continue
            names.append(feat_name)
            data = p.get_feat(feat_name, only_sleeptime=True,
                              wsize=300, step=30)
            data = data[:length]
            data = functions.interpolate_nans(data)

            # # first per stage
            feat_mean = [np.nanmean(data[hypno == stage]) if (hypno == stage).any() else np.nan for stage in stages]
            feat_std = [np.nanstd(data[hypno == stage]) if (hypno == stage).any() else np.nan for stage in stages]

            quantiles = list(range(0, 100, 25))
            feat_quantiles = np.array([[np.nanpercentile(data[hypno == stage], q) if (hypno == stage).any() else np.nan for stage in stages] for q in quantiles])
            feat_quantiles = flatten(feat_quantiles)
            feats += feat_mean + feat_std + feat_quantiles

            mods = ['mean', 'std'] + [f'quant{q}' for q in quantiles]
            for stage in stages:
                # , 'median', 'min', 'max', 'std']])
                feature_names.extend([f'{feat_name}_{mod}_stage{stage}' for mod in mods])

        data_x.append(feats)
        data_y.append(p.group == 'nt1')

    if len(feature_names) != len(feats):
        print(f'Names do not match length of feature vector {len(feats)}!={len(feature_names)}')

    data_x = np.array(data_x)
    # replace nan values with the mean of this value over all other participants
    for i in range(data_x.shape[-1]):
        data_x[np.isnan(data_x[:, i]), i] = np.nanmean(data_x[:, i])
    data_x = np.nan_to_num(data_x, posinf=0, neginf=0)
    data_y = np.array(data_y)

    # %% train
    clf = RandomForestClassifier(1000)
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
    name = 'RFC-feat-stages'
    misc.save_results(report, name, ss=ss, clf=clf)

    # %% feat select
    selector = RFECV(clf, cv=5, n_jobs=-1, verbose=100, step=1, scoring='f1')
    selector.fit(data_x, data_y)
    selector.grid_scores_

    print(f'These were the {selector.n_features_} features that were deemed important')
    print([feature_names[i] for i in np.nonzero(selector.support_)[0]])

    # %% feature importance analysis
    clf.fit(data_x, data_y)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names_sorted = [feature_names[i] for i in indices]

    plot_n = 20

    plt.bar(np.arange(plot_n), importances[indices][:plot_n])
    plt.xticks(np.arange(plot_n), feature_names_sorted[:plot_n], rotation=70)
    plt.subplots_adjust(bottom=0.6)
    plt.title('First 15 most important features for NT1 classification')
