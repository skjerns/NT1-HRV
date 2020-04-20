# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:59:02 2020

This script tries to replicate the findings by van Meijden (2015), 
https://doi.org/10.1111/jsr.12253

Their major finding was
    - Heart rate was significantly higher for NT1
    - Groups did not differ in heart rate variability measures.

Participants:
    - had not used and were not using medication for narcolepsy
    - gender&age matchted (+-5years), no medical conditions, not pregnant
    - no use of hypnotics or drugs affecting HR, a history of diabetes mellitus

Calculations:
    Cardiovascular:
    - the mean HR of each epoch was calculated
    - HR, HRV, resp. freq var. for each sleep stage 
    - low frequency (LF; 0.04–0.15 Hz) power via FFT (sympathetical activity)
    - high frequency (HF; 0.15–0.4 Hz) power via FFT (vagal activity)
    - total power 0–0.4 Hz range (up = decline in sympathetic tone)
    - resp. frequency from thoracic respiratory band movements
    - (HF^RF) was calculated by 0.65 x respiratory frequency (Hz)
    - (??^RF) upper limit by 1.35 x respiratory frequency (Hz)
    - the LF^RF/HF^RF ratio was calculated
    
    PSG:
    - total sleep time, 
    - sleep efficiency, 
    - latency to sleep stage 1, -
    - (REM) latency from sleep onset and 
    - stage shift index (number of shifts between sleep stages per hour) 
    - relative durations of stages 1–3 of total sleep time. 

    - HR, LF/HF, LF/HF^RF, total power of HRV, RF for sleep stages and WASO
    - epochs of sleep following onset of nocturnal sleep and 
      prior to awakening in the morning
     
     
Epochs included:
    - Median length of each sleep stage on group level, only epochs above 
    
    This study first identified all ‘periods’ of a sleep stage: this is a 
    series of consecutive epochs of a particular sleep stage. The duration 
    of such periods will likely vary considerably within and between subjects. 
    Including all epochs would result in a large number of epochs immediately 
    following a state shift and a progressively lower number of epochs as the 
    duration of the sleep stage becomes lower. To avoid this problem, we chose 
    a specific duration for each sleep stage. The method chosen to define that 
    period was based on the median duration of all periods of that stage. 
    In view of expected differences in duration between cases and controls, 
    this study calculated the median duration separately for the NC and the 
    control group. The shortest of both values was applied to both groups and 
    used for analysis. Periods with a shorter duration than the median value 
    were omitted from analysis, and periods with longer duration were included,
    but only the part until the defined cut-off value. This analysis could 
    result in differences in numbers of periods that subjects contributed 
    to the analysis, so the number of periods per subject was tabulated to 
    check for any such bias. As only relative minor interindividual differences
    were seen in this number (Table S1), we used the group median duration for
    the analysis. For each period the temporal order of epochs was recorded 
    (first, second, third, etc.) and included as the variable ‘epoch order’ 
    to assess the effects of sleep state duration.


Additionally:
    we labelled an epoch with ‘arousal transition’ when this epoch followed a 
    transition from SWS to sleep stage 2 or 1, and from sleep stage 2 to 1.
    Only epochs without apnoeas, leg movements (LM) and arousal transitions 
    were included in the analysis.


TODO: Remove artefact from sleep transition score (done?)
TODO: switch to Mann–Whitney U-test??

Notes:
    I'm taking the mean instead of median for episode/phase calculation
    If I'm taking the median as in the paper I get very low numbers

@author: Simon Kern
"""

import ospath
import stimer
import functions
from functions import arousal_transitions
import numpy as np
import config as cfg
import scipy.stats as stats
from sleep import Patient, SleepSet
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

stimer.start('All calculations')

ss = SleepSet(cfg.folder_unisens, readonly=True)
ss = ss.filter(lambda x: x.match!='') # only use matched participants
ss = ss.filter(lambda x: len(x.get_hypno())>0)
p = ss[1]

#%%### Recreate Table 1

descriptors = ['age', 'gender', 'TST', 'sleep efficiency', 'REM latency',
               'S1 ratio', 'S2 ratio', 'SWS ratio', 'REM ratio', 'WASO ratio',
               'Stage shift index', 'Arousals', 'Arousal transitions']

table1 = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)

    # gender
    table1['gender'][group]['female'] = np.sum([x.gender=='female' for x in subset])
    table1['gender'][group]['male'] = np.sum([x.gender=='male' for x in subset])

    # age
    values = np.array([x.age for x in subset])
    table1['age'][group]['values'] = values
    
    # TST
    values = np.array([np.sum((p.get_hypno()!=5)+(p.get_hypno()!=0)) for p in subset])
    table1['TST'][group]['values'] = values
    
    # Sleep efficiency
    values = np.array([len(p.get_hypno(only_sleeptime=True)) for p in subset])
    values = values / table1['TST'][group]['values']
    table1['sleep efficiency'][group]['values'] = values
    
    # REM latency
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)==4) for p in subset])
    table1['REM latency'][group]['values'] = values
    
    # Sleep stage 1 ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==1) for p in subset])
    table1['S1 ratio'][group]['values'] = values

     # Sleep stage 2 ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==2) for p in subset])
    table1['S2 ratio'][group]['values'] = values

     # SWS ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==3) for p in subset])
    table1['SWS ratio'][group]['values'] = values
    
     # REM ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==3) for p in subset])
    table1['REM ratio'][group]['values'] = values
    
    # Stage shift index
    hypnos = [p.get_hypno(only_sleeptime=True) for p in subset]
    values = np.array([np.count_nonzero(np.diff(hypno[hypno!=5])) for hypno in hypnos])
    values = values / np.array([len(p.get_hypno(only_sleeptime=True)) for p in subset])*120
    table1['Stage shift index'][group]['values'] = values
    
    # LM index (movements)
    # PLM index
    # we dont have these annotations, so skip
    
    # Arousal transitions
    
    hypnos = [p.get_hypno() for p in subset]
    arousals = [p.get_arousals() for p in subset]
    values = [arousal_transitions(h, a) for h, a in zip(hypnos, arousals)]
    table1['Arousal transitions'][group]['values'] = values
    
    #### Additional
    # WASO ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==0) for p in subset])
    values = values / table1['TST'][group]['values']
    table1['WASO ratio'][group]['values'] = values
    
    # Number of arousals
    values = [p.get_arousals() for p in subset]
    values = [len(v) for v in values]
    table1['Arousals'][group]['values'] = values
    
    
# calculate mean, std and p values for the values above
for descriptor in descriptors:
    if descriptor=='gender': continue
    values_nt1 = table1[descriptor]['nt1']['values']
    values_control = table1[descriptor]['control']['values']
    table1[descriptor]['p'] = stats.mannwhitneyu(values_nt1, values_control).pvalue
    for group in ['nt1', 'control']:
        table1[descriptor][group]['mean'] = np.mean(table1[descriptor][group]['values'])
        table1[descriptor][group]['std'] = np.std(table1[descriptor][group]['values'])
        
#####################################
#%% Calculate mean episode length // Recreate Figure 2

fig, axs = plt.subplots(2,2)

means = dict(zip(range(6), [{} for _ in range(6)]))

for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)
    phase_counts = list(map(functions.phase_lengths, subset.get_hypnos(only_sleeptime=True)))
    phase_counts_all = dict(zip(range(6), [[] for _ in range(6)]))
    for phase_count in phase_counts: 
        for i in phase_count:
            phase_counts_all[i].extend(phase_count[i])
            
    
    for stage in range(6):
        durations = np.array(phase_counts_all[stage], dtype=int)/2
        means[stage][group] = np.mean(durations)
        if stage in [0, 5]: continue
    
        hist = np.histogram(durations, bins=np.arange(16))
        ax = axs.flatten()[stage-1]
        ax.plot(hist[1][:-1], hist[0])
        ax.set_title(cfg.num2stage[stage])
        ax.set_xlabel('minutes')
        ax.set_ylabel('# of epochs')
        ax.legend(['nt1', 'control'])
        
plt.suptitle('Number of sleep phases for different phase lengths')
for i, stage in enumerate([1,2,3,4]):
    ax = axs.flatten()[i]
    minmean = min(means[stage]['nt1'], means[stage]['control'])
    ax.axvline(minmean, linestyle='dashed', linewidth=0.5, c='black')

#%% calculate metrics for different sleep stages: HF/LF, HRV, etc

features = ['LF', 'HF', 'LF_HF', 'mean_HR', 'RMSSD', 'pNN50', 'n_epochs']
table2 = {name:{stage:{'nt1':{}, 'control':{}} for stage in range(6)} for name in features}
masks = {'nt1':{}, 'control':{}}

for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl'))
    phase_counts = list(map(functions.phase_lengths, subset.get_hypnos(only_sleeptime=True)))
    phase_lengths = list(map(functions.stage2length, subset.get_hypnos(only_sleeptime=True)))
    
    
    for stage in range(5):
        ##### here we filter out which epochs are elegible for analysis
        # only take epochs that are longer than the mean epoch length
        minmean = min(means[stage]['nt1'], means[stage]['control'])
        mask_length = [np.array(l)>minmean for l in phase_lengths]
        # onlt take epochs of the given stage
        mask_stage = [h==stage for h in subset.get_hypnos(only_sleeptime=True)]
        # only take epochs with no artefacts surrounding 300 seconds
        mask_art = [p.get_artefacts(only_sleeptime=True, block_window_length=300)==False for p in subset]
        assert all([len(x)==len(y) for x,y in zip(mask_length, mask_stage)])
        assert all([len(x)==len(y)  for x,y in zip(mask_length, mask_art)])
        # create a signel mask
        mask = [np.logical_and.reduce((x,y,z)) for x,y,z in zip(mask_length, mask_stage, mask_art)]
        masks[group][stage] = mask
        print(f'{stage}: {np.mean([np.mean(x) for x in mask])}')
 
    


for feat in tqdm(features, desc='Calculating features'):
    
    for stage in range(5):
        for group in 'nt1', 'control':
            subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl'))
            # get all subset that have features
            feat_idx = cfg.feats_mapping[feat]
            # get all values
            values = [p.get_feat(feat_idx, only_sleeptime=True, cache=True) for p in subset]
            # combine values with masks
            values = [np.mean(v[m[:len(v)]]) for v, m in zip(values, masks[group][stage])]
            table2[feat][stage][group] = {'values':values}
            table2[feat][stage][group]['mean'] = np.nanmean(values)
            table2[feat][stage][group]['std'] = np.nanmean(values)
        values_nt1 = table2[feat][stage]['nt1']['values']
        values_control = table2[feat][stage]['control']['values']
        table2[feat][stage]['p'] = stats.mannwhitneyu(values_nt1, values_control).pvalue


#%%
stimer.stop('All calculations')
