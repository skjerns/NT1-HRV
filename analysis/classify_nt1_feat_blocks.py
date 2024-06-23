# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:58:00 2020

Detect NT1 when a scoring is available

@author: Simon Kern
"""
import sys; sys.path.append('..')
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
import misc
from natsort import natsort_key
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from misc import auc

np.random.seed(0)
flatten = lambda t: [item for sublist in t for item in sublist]
float2RGB = lambda f: (0,0.4+f/2,0)
plt.close('all')
misc.low_priority()

#%% settings

length = int(2 * 60 * 3.5) # first four hours
block_length = 2 * 30 # 60 minute blocks
blocks = [[x, x+block_length] for x in np.arange(0, length, block_length)]

ss = SleepSet(cfg.folder_unisens)
# ss = ss.stratify() # only use matched participants
ss = ss.filter(lambda x: x.duration < 60*60*11)  # only less than 14 hours
ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # no hypersomnikers
ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True)[:length])<0.25) # only low artefacts count <25%
p = ss[1]


name = f'RFC-feat-blocks-{block_length}min-n{len(ss)}-no-offset'

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


res = Parallel(10)(delayed(extract)(p) for p in tqdm(ss, desc='loading and windowing features'))

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
misc.save_results(data_y, y_pred, name, params=params, ss=ss, clf=clf, subfolder=name)

#%% feature importances
clf = RandomForestClassifier(2000, n_jobs=-1)

# create feature importances
clf.fit(data_x, data_y)
ranking = clf.feature_importances_

df_importances = pd.DataFrame()
for val, feat in zip(ranking, feature_names):
    start = feature_names[feat]['start']
    ftype = feature_names[feat]['type']
    df_tmp = pd.DataFrame({'Feature Name': ftype,
                           'Block': f'{start:03d}-{start+block_length:03d} min',
                           'Relative Importance': val}, index=[0])
    df_importances = pd.concat([df_importances, df_tmp], ignore_index=True)


df_importances = df_importances.sort_values(['Relative Importance'], ascending=False)


#%%
os.makedirs(f'{cfg.documents}/results/{name}/', exist_ok=True)
# plot importance across features
order = df_importances.groupby('Feature Name').mean().sort_values(['Relative Importance'], ascending=False).index
plt.figure(figsize=[14,10])
sns.barplot(data=df_importances, y='Feature Name', x='Relative Importance', order=order, orient='h')
plt.title(f'Relative feature importance over all blocks\n{ss=}')
plt.pause(0.1)
plt.tight_layout()
plt.savefig(f'{cfg.documents}/results/{name}/importance_blocks_across.png')

# plot importance across blocks
plt.figure(figsize=[14,10])
sns.barplot(data=df_importances, x='Block', y='Relative Importance')
plt.title(f'Relative importance of blocks\n{ss=}')
plt.pause(0.1)
plt.tight_layout()
plt.savefig(f'{cfg.documents}/results/{name}/feature_importance_blocks.png')

# plot importance across all
fig, axs = misc.make_fig(n_axs=len(df_importances.Block.unique()), bottom_plots=0)
df_importances = df_importances.sort_values('Block', key=natsort_key)
for i, (block, df_block) in enumerate(df_importances.groupby('Block')):
    ax = axs[i]
    sns.barplot(data=df_block, x='Feature Name', y='Relative Importance',
                order=order, orient='v', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'Relative feature importance within Block {i}: {block}')
misc.normalize_lims(axs[:len(df_importances.groupby('Block'))])
fig.suptitle(f'Feature importances within different blocks\n{ss=}')
plt.pause(0.1)
fig.tight_layout()
fig.savefig(f'{cfg.documents}/results/{name}/feature_importance_blocks_within.png')

stop


#%% find out which got classified wrong
matplotlib.use('cairo') # speeds up plotting as no figure is shown

for i in tqdm(np.where(data_y)[0]):
    fig = plt.figure(figsize=[10,12])

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
