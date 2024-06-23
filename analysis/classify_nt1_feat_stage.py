# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 when a scoring is available

@author: Simon Kern
"""
import sys; sys.path.append('..')
import os
import misc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from natsort import natsort_key
from sleep import SleepSet
import config as cfg
import misc
import functions
import features
from joblib.parallel import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from misc import get_auc
from sklearn.feature_selection import RFECV
np.random.seed(0)
misc.low_priority()
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

name = f'RFC-feat-stages-n{len(ss)}'
# %% load data
stages = np.arange(5)

def extract(p):
    feature_names = {}
    hypno = p.get_hypno(only_sleeptime=True)
    feats = []
    hypno = hypno[:length]

    # feature per sleepstage
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

auc = get_auc(data_y, y_pred[:,1])
name = f'auc{auc:.2f}-RFC-stages-n{len(ss)}'
os.makedirs(f'{cfg.documents}/results/{name}/', exist_ok=True)
misc.save_results(data_y, y_pred, name, ss=ss, clf=clf, subfolder=name)
# %% feat select
# selector = RFECV(clf, cv=5, n_jobs=-1, verbose=100, step=1, scoring='f1')
# selector.fit(data_x, data_y)
# selector.grid_scores_

# print(f'These were the {selector.n_features_} features that were deemed important')
# print([feature_names[i] for i in np.nonzero(selector.support_)[0]])

#%% feature importances
stages = ['Wake', 'S1', 'S2', 'SWS', 'REM', 'Art']
d = dict(enumerate(stages))

clf = RandomForestClassifier(2000, n_jobs=-1)
# create feature importances
clf.fit(data_x, data_y)
ranking = clf.feature_importances_

df_importances = pd.DataFrame()
for val, feat in zip(ranking, feature_names):
    stage = feature_names[feat]['stage']
    ftype = feature_names[feat]['type']
    df_tmp = pd.DataFrame({'Feature Name': ftype,
                           'Stage': d[stage],
                           'Relative Importance': val}, index=[0])
    df_importances = pd.concat([df_importances, df_tmp], ignore_index=True)

#%%

# plot importance across features
order = df_importances.groupby('Feature Name').mean().sort_values(['Relative Importance'], ascending=False).index
plt.figure(figsize=[14,10])
sns.barplot(data=df_importances, y='Feature Name', x='Relative Importance', order=order, orient='h')
plt.title(f'Relative feature importance over all stages\n{ss=}')
plt.pause(0.1)
plt.tight_layout()
plt.savefig(f'{cfg.documents}/results/{name}/importance_stages_across.png')

# plot importance across stages
plt.figure(figsize=[14,10])
sns.barplot(data=df_importances, x='Stage', y='Relative Importance')
plt.title(f'Relative importance of stages\n{ss=}')
plt.pause(0.1)
plt.tight_layout()
plt.savefig(f'{cfg.documents}/results/{name}/feature_importance_stages.png')

# plot importance across all
fig, axs = plt.subplots(2, 3)
axs = axs.flatten()

for i, (stage, df_stage) in enumerate(df_importances.groupby('Stage', sort=False)):
    ax = axs[i]
    sns.barplot(data=df_stage, x='Feature Name', y='Relative Importance',
                order=order, orient='v', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'Relative feature importance within Stage {i}: {stage}')

misc.normalize_lims(axs[:len(df_importances.groupby('Stage'))])
fig.suptitle(f'Feature importances within different stages\n{ss=}')
plt.pause(0.1)
fig.tight_layout()
fig.savefig(f'{cfg.documents}/results/{name}/feature_importance_stages_within.png')


#%% last but not least make a table with all results
