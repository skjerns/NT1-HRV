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
#%%
stages = np.arange(5)

x_train = []
y_train = []

for p in tqdm(ss):
    hypno = p.get_hypno(only_sleeptime=True)
    feats = []
    for feat_name in cfg.mapping_feats:
        if feat_name not in features.__dict__: continue
        data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30, offset=True)
        feat_mean = [np.nanmean(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat_min = [np.nanmin(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat_max = [np.nanmax(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat_std = [np.nanstd(data[hypno==stage]) if (hypno==stage).any() else np.nan for stage in stages]
        feat = feat_mean + feat_min + feat_max + feat_std
        feats.extend(feat)
    x_train.append(feats)
    y_train.append(p.group=='nt1')

x_train = np.nan_to_num(x_train, posinf=0, neginf=0)
y_train = np.array(y_train)


#%% Train
clf = RandomForestClassifier(1000)
x = cross_validate(clf, x_train, y_train, cv=10, scoring= ['accuracy', 'f1'], n_jobs=16, verbose=0)
clf.fit(x_train, y_train)