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
TODO: Arousal transitions!

@author: Simon Kern
"""

import ospath
import stimer
import functions
import numpy as np
import config as cfg
import scipy.stats as stats
from sleep import Patient, SleepSet
from tqdm import tqdm


ss = SleepSet(cfg.folder_unisens)
nt1 = ss.filter(lambda x: x.group=='nt1')
controls = ss.filter(lambda x: x.group=='control')
p = ss[3]
results = {}

#%%### Recreate Table 1

descriptors = ['age', 'gender', 'TST', 'sleep efficiency', 'REM latency',
               'S1 ratio', 'S2 ratio', 'SWS ratio', 'REM ratio', 'WASO ratio',
               'Stage shift index', 'Arousals']

table1 = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)

    # gender
    table1['gender'][group]['female'] = np.sum([x.gender=='female' for x in subset])
    table1['gender'][group]['male'] = np.sum([x.gender=='male' for x in nt1])

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
    # Arousal transitions
    
    
    #### Additional
    # WASO ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==0) for p in subset])
    values = values / table1['TST'][group]['values']
    table1['WASO ratio'][group]['values'] = values
    stimer.lapse()   
    
    # Number of arousals
    values = [p.arousals.get_data() for p in subset if hasattr(p, 'arousals_txt')]
    values = [len(v.split('\n'))-6 for v in values]
    table1['Arousals'][group]['values'] = values
    
    
    
for descriptor in descriptors:
    if descriptor=='gender': continue
    values_nt1 = table1[descriptor]['nt1']['values']
    values_control = table1[descriptor]['control']['values']
    table1[descriptor]['p'] = stats.ttest_ind(values_nt1, values_control).pvalue
    for group in ['nt1', 'control']:
        table1[descriptor][group]['mean'] = np.mean(table1[descriptor][group]['values'])
        table1[descriptor][group]['std'] = np.std(table1[descriptor][group]['values'])
        
#####################################
#%% Calculate mean episode length // Recreate Figure 2


for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)
    phase_counts = list(map(functions.epoch_lengths, subset.get_hypnos(only_sleeptime=True)))
    