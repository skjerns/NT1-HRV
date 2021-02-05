# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 when a scoring is available

@author: Simon Kern
"""
import sys
import os
import misc
import matplotlib
import pandas as pd
import seaborn as sns
from joblib.parallel import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sleep import SleepSet, Patient
import config as cfg
import functions
import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
np.random.seed(0)
flatten = lambda t: [item for sublist in t for item in sublist]
float2RGB = lambda f: (0,0.4+f/2,0)
plt.close('all')

#%% settings

length = int(2 * 60 * 3.5) # first four hours
block_length = 2 * 30 # 60 minute blocks
blocks = [[x, x+block_length] for x in np.arange(0, length, block_length)]
name = f'RFC-feat-blocks-{block_length}epochs-no-offset'

ss = SleepSet(cfg.folder_unisens)
# ss = ss.stratify() # only use matched participants
ss = ss.filter(lambda x: x.duration < 60*60*11)  # only less than 14 hours
ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # no hypersomnikers
ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True)[:length])<0.25) # only low artefacts count <25%
p = ss[1]

# %% load data
data_x = []
data_y = []
def extract(p):
    feature_names = {}
    names = []
    offset = 0#-np.random.randint(0, min(p.sleep_onset, 1500))
    p.sleep_onset_offset = offset
    feats = []
    # feature per sleepstage / block
    for feat_name in cfg.mapping_feats:
        if feat_name not in features.__dict__:
            continue
        names.append(feat_name)
        data = p.get_feat(feat_name, only_sleeptime=True, wsize=300, step=30, sleep_onset_offset=offset)

        data = data[:length]
        data = functions.interpolate_nans(data)

        # build data on temporal blocks
        feat_mean = [np.nanmean(data[start:end])  for start, end in blocks]
        feat_std = [np.nanstd(data[start:end]) for start, end in blocks]

        quartiles = list(range(0, 100, 25))
        feat_quartiles = np.array([[np.nanpercentile(data[start:end], q) for start, end in blocks]  for q in quartiles])
        feat_quartiles = flatten(feat_quartiles)
        feats += feat_mean + feat_std + feat_quartiles

        mods = ['mean', 'std'] + [f'quart{q}' for q in quartiles]
        for start, end in blocks:
            _names = [f'{feat_name}_{mod}_block{start}:{end}' for mod in mods]
            for _name in _names:
                feature_names[_name] = {'start':start, 'type': feat_name}
    return feats, feature_names


res = Parallel(16)(delayed(extract)(p) for p in tqdm(ss, desc='extracting features'))

feature_names = res[0][1]
feature_types = list(set([x["type"] for x in feature_names.values()]))
data_x = np.array([x for x,_ in res])
data_y = np.array([p.group=='nt1' for p in ss])


if len(feature_names) != len(data_x.T):
    print(f'Names do not match length of feature vector {len(data_x)}!={len(feature_names)}')


# replace nan values with the mean of this value over all other participants
for i in range(data_x.shape[-1]):
    data_x[np.isnan(data_x[:, i]), i] = np.nanmean(data_x[:, i])
data_x = np.nan_to_num(data_x, posinf=0, neginf=0)
data_y = np.array(data_y)

# %% train
clf = RandomForestClassifier(2000, n_jobs=4)
cv = StratifiedKFold(n_splits = 20, shuffle=True)

y_pred = cross_val_predict(clf, data_x, data_y, cv=cv, method='predict_proba', n_jobs=5, verbose=10)

params = f'{length=}, {block_length=}, {feature_types=}'
# misc.save_results(data_y, y_pred, name, params=params, ss=ss, clf=clf)
# stop
#%% feature importances
clf = RandomForestClassifier(2000, n_jobs=-1)
# create feature importances
clf.fit(data_x, data_y)
ranking = clf.feature_importances_

block_importances = np.zeros(len(blocks))
type_importances = np.zeros(len(feature_types))
for val, name in zip(ranking, feature_names):
    start = feature_names[name]['start']
    ftype = feature_names[name]['type']
    block_importances[start//block_length]+=val
    type_importances[feature_types.index(ftype)] += val

block_importances = block_importances-block_importances.min()
block_importances = block_importances/block_importances.max()

# type_importances = type_importances-type_importances.min()
# type_importances = type_importances/type_importances.max()
df = pd.DataFrame({'Feature Name':feature_types, 'Relative Importance':type_importances}).sort_values('Relative Importance', ascending=False)
plot = sns.barplot(data=df, y='Feature Name', x='Relative Importance', orient='h')
plt.title('Relative feature importance over all timeframes')
# for item in plot.get_xticklabels():
    # item.set_rotation(-90)
plt.tight_layout()

stop


#%% find out which got classified wrong
matplotlib.use('cairo') # speeds up plotting as no figure is shown

for i in tqdm(np.where(data_y)[0]):
    fig = plt.figure(figsize=[10,12], maximize=False)

    p = ss[i]
    score = y_pred[i,1]
    title = f'class probas: {y_pred[i]}'
    file = os.path.join(cfg.documents, 'plots',f'NT1_true_{block_length//2}min',f'{score:.3f}_{p.group}_'+ p.code+ '.png')
    if os.path.exists(file): continue
    os.makedirs(os.path.dirname(file), exist_ok=True)

    fig, axs = p.spectogram(channels=['eeg','ecg', 'rri'], title=title, saveas=False, ufreq=35, fig=fig)
    ax = axs[-1]
    block_limits = (np.ravel(blocks)*30)+p.sleep_onset
    ax.vlines(block_limits, ax.get_ylim()[0]+0.5,  ax.get_ylim()[1]-0.5, 'grey', linewidth=0.5)
    ax.axvspan(ax.get_xlim()[0], block_limits[0], facecolor='0.1', alpha=0.3)
    ax.axvspan(block_limits[-1], ax.get_xlim()[1], facecolor='0.1', alpha=0.3)
    for start, end in blocks:
        val = block_importances[start//block_length]
        start = (start*30)+p.sleep_onset
        end = (end*30)+p.sleep_onset
        ax.axvspan(start, end, facecolor=float2RGB(val), alpha=0.3)
    fig.savefig(file)
    plt.close(fig)


for i in tqdm(np.where(~data_y)[0]):
    fig = plt.figure(figsize=[10,12], maximize=False)
    p = ss[i]
    score = y_pred[i,1]
    title = f'class probas: {y_pred[i]}'
    file = os.path.join(cfg.documents, 'plots',f'CNT_true_{block_length//2}min',f'{score:.3f}_{p.group}_'+ p.code+ '.png')
    if os.path.exists(file): continue
    os.makedirs(os.path.dirname(file), exist_ok=True)
    fig, axs = p.spectogram(channels=['eeg', 'ecg', 'rri'], title=title, saveas=False, ufreq=35, fig=fig)
    ax = axs[-1]
    block_limits = (np.ravel(blocks)*30)+p.sleep_onset
    ax.vlines(block_limits, ax.get_ylim()[0]+0.5,  ax.get_ylim()[1]-0.5, 'grey', linewidth=0.5)
    ax.axvspan(ax.get_xlim()[0], block_limits[0], facecolor='0.1', alpha=0.3)
    ax.axvspan(block_limits[-1], ax.get_xlim()[1], facecolor='0.1', alpha=0.3)
    for start, end in blocks:
        val = block_importances[start//block_length]
        start = (start*30)+p.sleep_onset
        end = (end*30)+p.sleep_onset
        ax.axvspan(start, end, facecolor=float2RGB(val), alpha=0.2)
    fig.savefig(file)
    plt.close(fig)

