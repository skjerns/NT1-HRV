# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:17:34 2020

compare the features calculated by kubios and calculated by our own algorithm


@author: skjerns
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import config as cfg
from sleep import SleepSet
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.ndimage import median_filter, convolve
from scipy.ndimage.filters import gaussian_filter1d

os.makedirs(os.path.join(cfg.documents, 'reports', 'feat_comparison'), exist_ok=True)

ss = SleepSet(cfg.folder_unisens)
p = ss[16]
p.reset()
matfile =  dict(p.feats.get_data())
kubios = matfile['TimeVar']

def best_corr(kubios, feat1):
    """
    we correlate all features of kubios with this feature
    this is a somewhat sound way to check whether our feature has
    the best correlation with what is actually calculated
    """
    df = pd.DataFrame(columns=['Name', 'corr', 'data'])

    for feat2_name, feat2 in kubios.items():
        if abs(len(feat2)-len(feat1))>10: continue
        if np.isnan(feat2).all(): continue
        min_len = min(len(feat1), len(feat2))
        mean = np.nan_to_num(np.nanmean(feat2))

        feat2 = np.nan_to_num(feat2[:min_len], nan=mean)
        feat2 = scipy.stats.zscore(convolve(feat2, weights=np.ones(5)/5, mode='wrap'))
        feat2 = np.nan_to_num(feat2)

        corrcoef, pval = scipy.stats.pearsonr(feat1, feat2)

        corr = np.correlate(feat1, feat2)[0]

        df = df.append({'Name': feat2_name, 'corr':corr, 'data':feat2},
                       ignore_index=True)

    df = df.sort_values('corr', ascending=False)
    df = df.reset_index()
    top_cutoff = df['corr'][0]*0.95

    best = df.loc[df['corr']>top_cutoff]

    return best

#%% precompute features
kubios['LF_HF'] = kubios['LF_power']/kubios['HF_power']
kubios['pNN50'] = kubios['pNNxx']
kubios['SDNN'] = kubios['std_RR']
kubios['SD2_SD1'] = kubios['Poincare_SD2_SD1']
kubios['SD1'] = kubios['Poincare_SD1']
kubios['SD2'] = kubios['Poincare_SD2']

# these ones we don't have
kubios['SDSD'] = np.random.randint(-1,1,len(kubios['Artifacts']))
kubios['triangular_index'] = np.random.randint(-1,1,len(kubios['Artifacts']))


# kubios['SDSD'] = kubios['']
to_delete = ['min_HR', 'max_HR', 'pNNxx', 'std_HR', 'std_RR', 'Poincare_SD2_SD1',
             'Poincare_SD1', 'Poincare_SD2']
for delete_name in to_delete:
    if delete_name in kubios:
        del kubios[delete_name]

#%% Compare features

plt.close('all')

feat_names = ['mean_HR', 'mean_RR', 'RMSSD', 'HF_power', 'LF_power', 'VLF_power', 'LF_HF',
              'pNN50', 'SDSD', 'SDNN', 'SD1', 'SD2', 'SD2_SD1', 'SampEn', 'triangular_index',
              'PNSindex', 'SNSindex', 'ApEn', 'DFA_alpha1']

table = []
for name in feat_names:

    feat1 = p.get_feat(name, wsize=30, offset=False)
    feat2 = kubios[name]

    min_len = min(len(feat1), len(feat2))
    mean_feat1 = np.mean(feat1[np.isfinite(feat1)])
    mean_feat2 = np.mean(feat2[np.isfinite(feat2)])
    feat1 = np.nan_to_num(feat1[:min_len], nan=mean_feat1, posinf=mean_feat1)
    feat2 = np.nan_to_num(feat2[:min_len], nan=mean_feat2, posinf=mean_feat1)

    # smooth noise over 5 epochs
    feat1 = scipy.stats.zscore(convolve(feat1, weights=np.ones(5)/5, mode='wrap'), nan_policy='omit')
    feat2 = scipy.stats.zscore(convolve(feat2, weights=np.ones(5)/5, mode='wrap'), nan_policy='omit')

    # feat1_smooth = gaussian_filter1d(feat1, 1)
    # feat2_smooth = gaussian_filter1d(feat2, 1)

    corrcoef, pval = scipy.stats.pearsonr(feat1, feat2)
    best = best_corr(kubios, feat1)
    print(name in best['Name'].values, name, best['Name'].values)

    maxlen = 200 # only plot the first X epochs
    plt.figure()
    plt.plot(feat1[:maxlen], linewidth=1.5, alpha=0.5, c='b')
    plt.plot(feat2[:maxlen], linewidth=1.5, alpha=0.5, c='g')
    if not name in best['Name'].values:
        plt.plot(best['data'][0][:maxlen], linewidth=1.5, alpha=0.5, c='r')
    plt.legend([f'{name}', f'Kubios {name}', f'best correlation {best["Name"][0]}'])
    plt.title(f'{name} corr-coeff {corrcoef}, p = {pval}\nBest Kubios match: {best["Name"].values}')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(cfg.documents, 'reports', 'feat_comparison', f'feat_comp_{name}.png'))


    # table.append([name, corrcoef, pval])





df = pd.DataFrame(table, columns=['Feature', 'corr-coeff', 'p-val'])
df.to_csv(os.path.join(cfg.documents,'reports', 'feature_correlation_kubios_own.csv'))