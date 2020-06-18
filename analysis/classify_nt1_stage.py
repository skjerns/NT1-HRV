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
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate

plt.close('all')
ss = SleepSet(cfg.folder_unisens)
ss = ss.stratify() # only use matched participants
p = ss[1]
#%% load data
stages = np.arange(5)

x_train = []
y_train = []

for p in tqdm(ss):
    feature_names = []
    hypno = p.get_hypno(only_sleeptime=True)
    feats = []
    for feat_name in cfg.mapping_feats:
        if feat_name not in features.__dict__: continue
        # if feat_name=='detrend_fluctuation': continue
        data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30, offset=True)
        feat_mean = [np.nanmean(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat_min = [np.nanmin(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat_max = [np.nanmax(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat_std = [np.nanstd(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat = feat_mean + feat_min + feat_max + feat_std
        feats.extend(feat)
        for stage in stages:
            feature_names.extend([f'{feat_name}_{mod}_stage{stage}' for mod in ['mean', 'min', 'max', 'std']])
    x_train.append(feats)
    y_train.append(p.group=='nt1')

x_train = np.array(x_train)
# replace nan values with the mean of this value over all other participants
for i in range(x_train.shape[-1]):
    x_train[np.isnan(x_train[:,i]), i] = np.nanmean(x_train[:,i])
x_train = np.nan_to_num(x_train, posinf=0, neginf=0)
y_train = np.array(y_train)


#%% train
clf = RandomForestClassifier(1000)
x = cross_validate(clf, x_train, y_train, cv=10, scoring= ['accuracy', 'f1'], n_jobs=16, verbose=0)
print(f'accuracy {np.mean(x["test_accuracy"]):.3f},\nf1 {np.mean(x["test_f1"]):.3f}')


#%% feature importance analysis
clf.fit(x_train, y_train)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
feature_names_sorted = [feature_names[i] for i in indices]

plt.bar(np.arange(15),importances[indices][:15])
plt.xticks(np.arange(15), feature_names_sorted[:15],rotation=90)
plt.subplots_adjust(bottom=0.6)
plt.title('First 15 most important features for NT1 classification')
