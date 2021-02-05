# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:43:29 2021

@author: Simon
"""
import os
import matplotlib
import features
import functions
import config as cfg
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sleep import SleepSet
from datetime import timedelta
import matplotlib.pyplot as plt
from joblib.parallel import Parallel, delayed

#%% settings

length = int(2 * 60 * 3.50) # first four hours

ss = SleepSet(cfg.folder_unisens)
ss = ss.filter(lambda x: x.duration < 60*60*11)  # only less than 14 hours
ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # no hypersomnikers
ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True)[:length])<0.25) # only low artefacts count <25%
p = ss[1]
data_y = np.array([p.group=='nt1' for p in ss], dtype=bool)

stop
#%% average hypnograms to see importances
hypnos = np.array([p.get_hypno(only_sleeptime=True)[:length+1] for p in ss])
cmap = matplotlib.cm.get_cmap("magma", 6)
hypno_nt1 = hypnos[data_y]
hypno_cnt = hypnos[~data_y]
hypno_nt1[hypno_nt1==5] = -1 # replace artefact with 0
hypno_cnt[hypno_cnt==5] = -1 # replace artefact with 0
hypno_nt1.sort(0)
hypno_cnt.sort(0)
fig, axs = plt.subplots(2, 1, gridspec_kw={'hspace':0.1})
img = axs[0].imshow(hypno_cnt, cmap, aspect='auto')
img = axs[1].imshow(hypno_nt1, cmap, aspect='auto')
axs[0].set_ylabel('Controls')
axs[1].set_ylabel('NT1 patients')
axs[1].set_xlabel('Hours after sleep onset')
axs[0].set_xticks(np.arange(0,len(hypno_cnt.T), 60))
axs[1].set_xticks(np.arange(0,len(hypno_cnt.T), 60))
axs[0].xaxis.set_major_formatter(lambda x,y:str(timedelta(minutes=x/2))[:-3])
axs[1].xaxis.set_major_formatter(lambda x,y:str(timedelta(minutes=x/2))[:-3])

cbar1 = fig.colorbar(img, ax=axs[0])
cbar2 = fig.colorbar(img, ax=axs[1])
cbar1.set_ticks(np.arange(-1, 5)*0.83+0.2)
cbar2.set_ticks(np.arange(-1, 5)*0.83+0.2)

cbar1.set_ticklabels(['A', 'W', 'S1', 'S2', 'SWS', 'REM'])
cbar2.set_ticklabels(['A', 'W', 'S1', 'S2', 'SWS', 'REM'])
axs[0].set_title('Distribution of sleep stages for both groups')


#%% plot average feature progression


def plot(feat_name, ss):

    matplotlib.use('cairo')
    data_x_all = np.array([feat[:length+1] for feat in ss.get_feats(feat_name, only_sleeptime=True)])
    data_x_all = functions.interpolate_nans(data_x_all)
    if feat_name=='KatzFract': #manual outlier removal
        data_x_all[data_x_all<-500] = np.mean(data_x_all[data_x_all>-500] )

    data_y_all = np.array([p.group=='nt1' for p in ss])


    fig = plt.figure(figsize=[10,12])
    axs = fig.subplots(4, 2, squeeze=True).flatten()
    for i, ax in enumerate(axs):
        if i<len(axs)-1:
            idx = np.random.choice(np.where(~data_y_all)[0], data_y_all.sum(), replace=False)
            idx = np.hstack([idx, np.where(data_y_all)[0]])
            data_x = data_x_all[idx]
            data_y = data_y_all[idx]
            title = f'Bootstrap {i+1}'
        else:
            title = 'All samples'
            data_x = data_x_all
            data_y = data_y_all

        groups = np.repeat([f'NT1, n={data_y.sum()}' if x else f'Control, n={(~data_y).sum()}' for x in data_y], len(data_x.T))
        df = pd.DataFrame({'value + SD':data_x.ravel(),
                           'epoch': list(range(len(data_x.T)))*len(data_x),
                           'groups':groups})

        sns.lineplot(data=df, x='epoch', y='value + SD', hue='groups', ci='sd', legend="full", ax=ax)
        ax.set_title(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Progression of {feat_name} after sleep onset for same-sized subsamples')
    plt.savefig(os.path.join(cfg.documents, 'plots', f'feature_{feat_name}_progression.png'))

feat_names = list(set(cfg.mapping_feats).intersection(set(features.__dict__)))
Parallel(8)(delayed(plot)(name, ss) for name in tqdm(feat_names, desc='creating plots'))

