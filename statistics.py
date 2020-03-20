# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:09:10 2020

@author: skjerns
"""
import config
import ospath
from sleep import SleepSet, Patient
import seaborn as sns
import scipy
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    folders = ospath.list_folders('z:/NT1-HRV-data')
    ss = SleepSet(folders)
    print(repr(ss))
    print(repr(ss[-1]))
    ss = ss.filter(lambda x: 'hypnogram' in x)
    controls = ss.filter(lambda x: x.group=='control')
    nt1 = ss.filter(lambda x: x.group=='nt1')
    hypno_controls = controls.get_hypnos()
    hypno_nt1 = nt1.get_hypnos()

    stages_controls = [np.histogram(h, bins=[0,1,2,3,4,5, 6])[::-1] for h in hypno_controls]
    stages_nt1 = [np.histogram(h, bins=[0,1,2,3,4,5, 6])[::-1] for h in hypno_nt1]

    distr_controls = np.vstack([stage/stage.sum() for bin, stage in stages_controls]).T
    distr_nt1 = np.vstack([stage/stage.sum() for bin, stage in stages_nt1]).T


    
    ax = plt.subplots(2,3, squeeze=True)[1].ravel()
    plt.suptitle('Comparison of stage/recording')
    for stage in range(6):
        plt.sca(ax[stage])
        p = scipy.stats.ttest_ind_from_stats(distr_controls[stage,:], distr_nt1[stage,:])
        sns.distplot(distr_controls[stage,:], bins=10)
        sns.distplot(distr_nt1[stage,:], bins=10)
        plt.legend(['Control', 'NT1'])
        plt.title(f'{config.num2stage[stage]}, p={p.pvalue:.4f}')
        plt.xlabel('percentage/bedtime')
        plt.ylabel('amount')
        
        
    hypno_controls = controls.get_hypnos(only_sleeptime=True)
    hypno_nt1 = nt1.get_hypnos(only_sleeptime=True)

    stages_controls = [np.histogram(h, bins=[0,1,2,3,4,5, 6])[::-1] for h in hypno_controls]
    stages_nt1 = [np.histogram(h, bins=[0,1,2,3,4,5, 6])[::-1] for h in hypno_nt1]

    distr_controls = np.vstack([stage/stage.sum() for bin, stage in stages_controls]).T
    distr_nt1 = np.vstack([stage/stage.sum() for bin, stage in stages_nt1]).T 
        
        
    ax = plt.subplots(2,3, squeeze=True)[1].ravel()
    plt.suptitle('Comparison of stage/sleeptime')
    for stage in range(6):
        plt.sca(ax[stage])
        p = scipy.stats.ttest_ind_from_stats(distr_controls[stage,:], distr_nt1[stage,:])
        sns.distplot(distr_controls[stage,:], bins=10)
        sns.distplot(distr_nt1[stage,:], bins=10)
        plt.legend(['Control', 'NT1'])
        plt.title(f'{config.num2stage[stage]}, p={p.pvalue:.4f}')
        plt.xlabel('percentage/bedtime')
        plt.ylabel('amount')