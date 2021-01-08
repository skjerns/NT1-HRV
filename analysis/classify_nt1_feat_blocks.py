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
    ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # no hypersomnikers
    ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True))<0.25) # only low artefacts count <25%
    p = ss[1]

    # %% load data
    length = 2 * 60 * 4 # first four hours
    block_length = 2 * 15 # 60 minute blocks
    blocks = [[x, x+block_length] for x in np.arange(0, length, block_length)]

    data_x = []
    data_y = []

    for p in tqdm(ss, desc='Loading features'):
        feature_names = []
        names = []
        hypno = p.get_hypno(only_sleeptime=True)
        feats = []
        hypno = hypno[:length]

        # feature per sleepstage / block
        for feat_name in cfg.mapping_feats:
            if feat_name not in features.__dict__:
                continue
            names.append(feat_name)
            data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30)
            data = data[:length]
            data = functions.interpolate_nans(data)

            # build data on temporal blocks
            feat_mean = [np.nanmean(data[start:end])  for start, end in blocks]
            feat_std = [np.nanstd(data[start:end]) for start, end in blocks]

            quantiles = list(range(0, 100, 25))
            feat_quantiles = np.array([[np.nanpercentile(data[start:end], q) for start, end in blocks]  for q in quantiles])
            feat_quantiles = flatten(feat_quantiles)
            feats += feat_mean + feat_std + feat_quantiles

            mods = ['mean', 'std'] + [f'quant{q}' for q in quantiles]
            for start, end in blocks:
                block = f'{start}:{end}'
                feature_names.extend([f'{feat_name}_{mod}_block{block}' for mod in mods])

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
    name = f'RFC-feat-blocks-{block_length}'
    misc.save_results(report, name, ss=ss, clf=clf)
