# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 when a scoring is available

@author: Simon Kern
"""
import sys
import os
import misc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sleep import SleepSet
import config as cfg
import functions
import features
from joblib.parallel import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
np.random.seed(0)

def flatten(t): return [item for sublist in t for item in sublist]

#%%
length = int(2 * 60 * 3.5) # first 3.5 hours

plt.close('all')
ss = SleepSet(cfg.folder_unisens)
# ss = ss.stratify() # only use matched participants
ss = ss.filter(lambda x: x.duration < 60*60*11)  # only less than 14 hours
ss = ss.filter(lambda x: x.group in ['control', 'nt1'])
ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True)[:length])<0.25)
p = ss[1]
name = 'RFC-feat-stages'
# %% load data
stages = np.arange(5)

def extract(p):
    feature_names = {}
    hypno = p.get_hypno(only_sleeptime=True)
    feats = []
    hypno = hypno[:length]

    # feature per sleepstage / block
    for feat_name in cfg.mapping_feats:
        if feat_name not in features.__dict__:
            continue
        data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30)
        data = data[:length]
        data = functions.interpolate_nans(data)

        # # first per stage
        feat_mean = [np.nanmean(data[hypno == stage]) if (hypno == stage).any() else np.nan for stage in stages]
        feat_std = [np.nanstd(data[hypno == stage]) if (hypno == stage).any() else np.nan for stage in stages]

        quartiles = list(range(0, 100, 25))
        feat_quantiles = np.array([[np.nanpercentile(data[hypno == stage], q) if (hypno == stage).any() else np.nan for stage in stages] for q in quartiles])
        feat_quantiles = flatten(feat_quantiles)
        feats += feat_mean + feat_std + feat_quantiles

        mods = ['mean', 'std'] + [f'quart{q}' for q in quartiles]
        for stage in stages:
            _names = [f'{feat_name}_{mod}_stage{stage}' for mod in mods]
            for _name in _names:
                feature_names[_name] = {'stage':stage, 'type': feat_name}

    return feats, feature_names
res = Parallel(16)(delayed(extract)(p) for p in tqdm(ss, desc='extracting features'))

feature_names = res[0][1]
feature_types = list(set([x["type"] for x in feature_names.values()]))
data_x = np.array([x for x,_ in res])
data_y = np.array([p.group=='nt1' for p in ss])

if len(feature_names) != len(data_x.T):
    print(f'Names do not match length of feature vector {len(data_x.T)}!={len(feature_names)}')

data_x = np.array(data_x)
# replace nan values with the mean of this value over all other participants
for i in range(data_x.shape[-1]):
    data_x[np.isnan(data_x[:, i]), i] = np.nanmean(data_x[:, i])
data_x = np.nan_to_num(data_x, posinf=0, neginf=0)
data_y = np.array(data_y)

# %% train
clf = RandomForestClassifier(2000, n_jobs=4)
cv = StratifiedKFold(n_splits = 20, shuffle=True)

y_pred = cross_val_predict(clf, data_x, data_y, cv=cv, method='predict_proba', n_jobs=4, verbose=10)

misc.save_results(data_y, y_pred, name, ss=ss, clf=clf)

stop
# %% feat select
selector = RFECV(clf, cv=5, n_jobs=-1, verbose=100, step=1, scoring='f1')
selector.fit(data_x, data_y)
selector.grid_scores_

print(f'These were the {selector.n_features_} features that were deemed important')
print([feature_names[i] for i in np.nonzero(selector.support_)[0]])

#%% feature importances
clf = RandomForestClassifier(2000, n_jobs=-1)
# create feature importances
clf.fit(data_x, data_y)
ranking = clf.feature_importances_

type_importances = np.zeros(len(feature_types))
stage_importances = np.zeros(len(stages))
for val, name in zip(ranking, feature_names):
    stage = feature_names[name]['stage']
    ftype = feature_names[name]['type']
    stage_importances[stage] += val
    type_importances[feature_types.index(ftype)] += val

df = pd.DataFrame({'Feature Name':feature_types, 'Relative Importance':type_importances}).sort_values('Relative Importance', ascending=False)
plot = sns.barplot(data=df, y='Feature Name', x='Relative Importance', orient='h')
plt.title('Relative feature importance over all timeframes')
# for item in plot.get_xticklabels():
    # item.set_rotation(-90)
plt.tight_layout()
plt.figure()
df = pd.DataFrame({'Sleep Stage':['W', 'S1', 'S2', 'SWS', 'REM'], 'Relative Importance':stage_importances})
plot = sns.barplot(data=df, x='Sleep Stage', y='Relative Importance')
plt.title('importance of sleep stages for prediction')
# for item in plot.get_xticklabels():
    # item.set_rotation(-90)
plt.tight_layout()
