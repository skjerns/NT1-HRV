# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:44:21 2020

Trying to classify sleep stages with HRV features.
Let's see if this works!

@author: Simon Kern
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import stimer
import functions
from functions import arousal_transitions
import numpy as np
import config as cfg
import scipy
import scipy.signal as signal
from sleep import SleepSet
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotting
import ospath
from itertools import permutations
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate


plt.close('all')
ss = SleepSet(cfg.folder_unisens)
ss = ss.stratify() # only use matched participants
p = ss[1]

#%% step 1: get features
if __name__=='__main__':

    train_x = []
    train_y = []

    for p in tqdm(ss, desc='Loading features'):
        feature_names = []
        feats = []
        stages = []
        hypno = p.get_hypno()

        for feat_name in cfg.mapping_feats:
            if feat_name not in features.__dict__: continue
            data = p.get_feat(feat_name, wsize=30, step=30, offset=True)
            assert len(data) == len(hypno)
            feats.append(data)

        train_x.extend(np.array(feats).T)
        train_y.extend(hypno)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # replace nan values with the mean of this value over all other participants
    for i in range(train_x.shape[-1]):
        train_x[~np.isfinite(train_x[:,i]), i] = np.nanmedian(train_x[:,i])

#%% ML

clf = RandomForestClassifier(1000)
x = cross_validate(clf, train_x, train_y, cv=5, scoring= ['accuracy', 'f1_macro'], n_jobs=4, verbose=100)
print(f'accuracy {np.mean(x["test_accuracy"]):.3f},\nf1 {np.mean(x["test_f1_macro"]):.3f}')
