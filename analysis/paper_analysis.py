# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:59:02 2020

This script tries to replicate the findings by van Meijden (2015), 
https://doi.org/10.1111/jsr.12253

additionally, explorative analysis are made

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
TODO: Check NT1 N1 ratio >50%????
TODO: Definition of 'stable sleep' in Pizza
TODO: Remove artefacts from phase change HR/LF/HF (fig 3/4)

Calculations:
    - nr. of awakenings, average length of awakening: T.Roth 10.5664/jcsm.3004
    

Notes:
    I'm taking the mean instead of median for episode/phase calculation
    If I'm taking the median as in the paper I get very low numbers

@author: Simon Kern
"""
import sys
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


plt.close('all')

stimer.start('All calculations')
files = ospath.list_folders(cfg.folder_unisens)
stimer.start()
ss = SleepSet(files, readonly=True)
stimer.stop()
# ss = ss.filter(lambda x: 'body' in x) # only take patients with body position sensors
# ss = ss.stratify() # only use matched participants
p = ss[1]
stop
#%%### Van Meijden 2015 Table 1

descriptors = ['gender', 'age' , 'TST', 'sleep efficiency', 'S1 latency' ,
               'S2 latency','SWS latency','REM latency',
               'S1 ratio', 'S2 ratio', 'SWS ratio', 'REM ratio', 'WASO ratio',
               'Stage shift index', 'Arousals', 'Arousal transitions',
               'Nr. of Awakenings', 'Awakenings/Hour', 'Awakening Lengths']

table1 = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)

    # gender
    table1['gender'][group]['female'] = np.sum([x.gender=='female' for x in subset])
    table1['gender'][group]['male'] = np.sum([x.gender=='male' for x in subset])
    table1['gender'][group]['values'] = [[x.gender=='male' for x in subset]]
    table1['gender'][group]['mean'] = None
    table1['gender'][group]['std'] = None
    table1['gender'][group]['p'] = None

    # age
    values = np.array([x.age for x in subset])
    table1['age'][group]['values'] = values
    
    # TST in minutes
    values = np.array([len([x for x in p.get_hypno() if x in [1,2,3,4]])/2 for p in subset])
    table1['TST'][group]['values'] = values
    
    # Sleep efficiency
    values = np.array([len(p.get_hypno(only_sleeptime=True)) for p in subset])
    values =  table1['TST'][group]['values'] / values*2
    table1['sleep efficiency'][group]['values'] = values
   
    # S1 latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)==1)/2 for p in subset])
    table1['S1 latency'][group]['values'] = values
   
    # S2 latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)==2)/2 for p in subset])
    table1['S2 latency'][group]['values'] = values
    
    # SWS latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)==3)/2 for p in subset])
    table1['SWS latency'][group]['values'] = values
    
    # REM latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)==4)/2 for p in subset])
    table1['REM latency'][group]['values'] = values
    
    # Sleep stage 1 ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==1) for p in subset])
    values = values / (table1['TST'][group]['values']*2)
    table1['S1 ratio'][group]['values'] = values

     # Sleep stage 2 ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==2) for p in subset])
    values = values / (table1['TST'][group]['values']*2)
    table1['S2 ratio'][group]['values'] = values

     # SWS ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==3) for p in subset])
    values = values / (table1['TST'][group]['values']*2)
    table1['SWS ratio'][group]['values'] = values
    
     # REM ratio
    values = np.array([np.sum(p.get_hypno(only_sleeptime=True)==4) for p in subset])
    values = values / (table1['TST'][group]['values']*2)
    table1['REM ratio'][group]['values'] = values
    
    # Stage shift index
    hypnos = [p.get_hypno(only_sleeptime=True) for p in subset]
    values = np.array([np.count_nonzero(np.diff(hypno[hypno!=5])) for hypno in hypnos])
    values = values / (np.array([len(p.get_hypno(only_sleeptime=True)) for p in subset])/120)
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

    # Number of awakenings
    values = list([len(functions.phase_lengths(h)[0]) for h in subset.get_hypnos(only_sleeptime=True)])
    table1['Nr. of Awakenings'][group]['values'] = values
    
    # Number of awakenings//hour
    values = list([len(functions.phase_lengths(h)[0])/(len(h)/120) for h in subset.get_hypnos(only_sleeptime=True)])
    table1['Awakenings/Hour'][group]['values'] = values
    
    # Length of awakenings
    values = list([np.mean(functions.phase_lengths(h)[0])/2 for h in subset.get_hypnos(only_sleeptime=True)])
    table1['Awakening Lengths'][group]['values'] = values
  
    
  

# calculate mean, std and p values for the values above
functions.calc_statistics(table1)
# plot distributions for the two groups and save results to html
plotting.print_table(table1, 'Sleep Stage Parameters') 
table1.pop('gender') # remove this, can't be plotted appropriately
plotting.distplot_table(table1, 'Sleep Stage Parameters', ylabel='counts')

#%% Van Meijden 2015 Figure 2, Mean episode length
stage_means = dict(zip(range(6), [{'control':{}, 'nt1':{}} for _ in range(6)]))
fig, axs = plt.subplots(2, 2)

for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl'))

    phase_counts = list(map(functions.phase_lengths, subset.get_hypnos(only_sleeptime=True)))
    phase_counts_all = dict(zip(range(6), [[] for _ in range(6)]))
    for phase_count in phase_counts: 
        for i in phase_count:
            phase_counts_all[i].extend(phase_count[i])
            
    
    for stage in range(6):
        durations = np.array(phase_counts_all[stage], dtype=int)/2
        stage_means[stage][group]['values'] = durations
        if stage in [0, 5]: continue
    
        hist = np.histogram(durations, bins=np.arange(16))
        ax = axs.flatten()[stage-1]
        ax.plot(hist[1][:-1], hist[0])
        ax.set_title(cfg.num2stage[stage])
        ax.set_xlabel('minutes')
        ax.set_ylabel('total # of epochs')
        ax.legend(['nt1', 'control'])
n = len(ss.filter(lambda x: hasattr(x, 'feats.pkl') and not x.ecg_broken=='False'))
plt.suptitle(f'Number of sleep phases for different phase lengths, n={n}')

# calculate statistics and print out results
functions.calc_statistics(stage_means)
plotting.print_table(stage_means, 'Number of sleep phases for different phase lengths') 

for i, stage in enumerate([1,2,3,4]):
    ax = axs.flatten()[i]
    minmean = min(stage_means[stage]['nt1']['mean'], stage_means[stage]['control']['mean'])
    ax.axvline(minmean, linestyle='dashed', linewidth=2, c='black')
    
#%% Van Meijden 2015 Figure 3: Feat Change after transition

features = ['mean_HR', 'LF', 'HF', 'LF_HF']
post_transitions = {feat:{stage:{} for stage in range(6)} for feat in features} 
   
for name in features:
    for stage in range(6):
        for group in ['nt1', 'control']:
            subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl') and x.ecg_broken==False)
            values = []
            
            for p in subset:
                hypno = p.get_hypno(only_sleeptime=True)
                # get the phase sequence, the lengths of the phase, 
                # and their start in the hypnogram
                stages, lengths, idxs = functions.sleep_phase_sequence(hypno)
                # get all indices where we have the given stage with longer
                # period than the mean stage length
                minmean = min(stage_means[stage]['nt1']['mean'], stage_means[stage]['control']['mean'])
                # we need it in epoch notation, so *2
                minmean = int(minmean*2)
                minmean = 16
                phase_starts = np.where(np.logical_and(stages==stage, lengths>minmean))[0]
                feat = p.get_feat(name, only_sleeptime=True)
                artefacts = p.get_artefacts(only_sleeptime=True)
                for start in phase_starts:
                    bool_idx = np.zeros(len(feat), dtype=bool)
                    bool_idx[idxs[start]: idxs[start]+lengths[start]] = True
                    feat_values = feat[idxs[start]: idxs[start]+lengths[start]]
                    artefacts = artefacts[idxs[start]: idxs[start]+lengths[start]]
                    values.append(feat_values[:minmean]) # only take %minmean% epochs
                    
            post_transitions[name][stage][group] = {'values':np.array(values)}
            
    # calculate statistics and print out results
    title = f'{name} rate changes directly after Sleep Phase Change'
    functions.calc_statistics(post_transitions[name])
    n = len(ss.filter(lambda x: hasattr(x, 'feats.pkl')))
    fig, axs = plotting.lineplot_table(post_transitions[name], title, xlabel='epochs', 
                                       ylabel=name, n=n)
    plotting.print_table(post_transitions[name], title) 

################################################
################################################

#%% Pizza 2015: transition indices + own indices
################################################
################################################

descriptors = ['Starting Sequences \nN1-N2-SWS-REM / N2-REM / SOREM / Other', 
               'Starting Sequences \nSWS-REM / N2-REM / SOREM',
               'Sequences W-S', 'Sequences W-NR-R', 'Sequences N1-NR-R']

pizza2015 = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)

    # Sleep Stage Sequences (see Pizza 2015)
    values = list([functions.starting_sequence_pizza(h) for h in subset.get_hypnos(only_sleeptime=True)])
    pizza2015['Starting Sequences \nN1-N2-SWS-REM / N2-REM / SOREM / Other'][group]['values'] = values
  
    # Sleep Stage Sequences (own definition, see functions.py)
    values = list([functions.starting_sequence(h) for h in subset.get_hypnos(only_sleeptime=True)])
    pizza2015['Starting Sequences \nSWS-REM / N2-REM / SOREM'][group]['values'] = values
    
    # Sleep Stage Sequences tW-Si (see Pizza 2015)
    values = np.array(list([functions.transition_index(h, [0]) for h in subset.get_hypnos()]))
    values = values / table1['TST'][group]['values']*60
    pizza2015['Sequences W-S'][group]['values'] = values
    
    # Sleep Stage Sequences tW-NR-Ri (see Pizza 2015)
    values = np.array(list([functions.transition_index(h, [0,1,4]) for h in subset.get_hypnos()]))
    values += np.array(list([functions.transition_index(h, [0,2,4]) for h in subset.get_hypnos()]))
    values = values / table1['TST'][group]['values']*60
    pizza2015['Sequences W-NR-R'][group]['values'] = values
    
    # Sleep Stage Sequences N1-NR-Ri (see Pizza 2015)
    values = np.array(list([functions.transition_index(h, [0,1,2,4]) for h in subset.get_hypnos()]))
    values += np.array(list([functions.transition_index(h, [0,1,3,4]) for h in subset.get_hypnos()]))
    values = values / table1['TST'][group]['values']*60
    pizza2015['Sequences N1-NR-R'][group]['values'] = values
    
# calculate mean, std and p values for the values above
functions.calc_statistics(pizza2015)

# plot distributions for the two groups and print out table
plotting.print_table(pizza2015, 'Sleep transition indices' ) 
plotting.distplot_table(pizza2015, 'Sleep transition indices', columns=2, 
                        xlabel=[*['Nr of Starting Sequence']*2, *3*['Nr of occurence / h']])


#%% Transition possibilities and compare between groups
descriptors = list(permutations([0,1,2,4], 2)) # all state transitions that are possible
descriptors.extend([(2,3), (3,2)])

transitions = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)
    for pair in descriptors:
        values = [len(functions.search_sequences(p.get_hypno(), np.array(pair))) for p in subset]
        nr_transitions = [np.count_nonzero(np.diff(p.get_hypno(only_sleeptime=True))) for p in subset]
        transitions[pair][group]['values'] = np.array(values)/nr_transitions
        
        # comment this in for transitions / hour
        # tst = [len(p.get_hypno(only_sleeptime=True))/120 for p in subset]
        # transitions[pair][group]['values'] = np.array(values)/np.array(tst)
       
        # comment this in for total number of transitions
        # transitions[pair][group]['values'] = np.array(values)/np.array(tst)
         
       
        
# calculate mean, std and p values for the values above
functions.calc_statistics(transitions)
plotting.distplot_table(transitions, '% of transitions of total transitions', columns=3, 
                        xlabel='% of all transitions', ylabel='count')
plotting.print_table(transitions, '% of transitions of total transitions' ) 

#%% Features for different sleep stages: HF/LF, HRV, etc

features = ['LF', 'HF', 'LF_HF', 'mean_HR', 'RMSSD', 'pNN50']
table2 = {name:{stage:{'nt1':{}, 'control':{}} for stage in range(5)} for name in features}
masks = {'nt1':{}, 'control':{}}

for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl') and x.ecg_broken==False)
    phase_counts = list(map(functions.phase_lengths, subset.get_hypnos(only_sleeptime=True)))
    phase_lengths = list(map(functions.stage2length, subset.get_hypnos(only_sleeptime=True)))
    
    
    for stage in range(5):
        ##### here we filter out which epochs are elegible for analysis
        # only take epochs that are longer than the mean epoch length
        minmean = min(stage_means[stage]['nt1']['mean'], stage_means[stage]['control']['mean'])
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
    if feat=='n_epochs':continue
    for stage in range(5):
        for group in 'nt1', 'control':
            subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl') and x.ecg_broken==False)
            # get all values
            values = [p.get_feat(cfg.mapping_feats[feat], only_sleeptime=True, cache=True) for p in subset]
            # combine values with masks
            values = [np.mean(v[m[:len(v)]]) for v, m in zip(values, masks[group][stage])]
            table2[feat][stage][group] = {'values':values}

functions.calc_statistics(table2)
plotting.print_table_with_subvars(table2, 'feats')


# plot feature distribution
# n_plots = len(descriptors[1:])
# fig, axs = plt.subplots(int(np.ceil(n_plots/3)), 3)
# axs = axs.flatten()
# for i, descriptor in enumerate(descriptors): 
#     ax = axs[i]
#     values_nt1 = table1[descriptor]['nt1']['values']
#     values_cnt = table1[descriptor]['control']['values']
#     sns.distplot(values_nt1, ax=ax)
#     sns.distplot(values_cnt, ax=ax)
#     p = plotting.format_p_value(table1[descriptor]['p'], bold=False)
#     ax.set_title(descriptor + f' p {p}')
#     ax.legend(['NT1', 'Control'])
    
# plt.suptitle('Overview Sleep Parameter Distribution')
# plt.tight_layout()


#########################
#%% Body position changes
#########################

body_positions = [cfg.mapping_body[x] for x in range(1,7)]
table_body = {name:{stage:{'nt1':{}, 'control':{}} for stage in range(5)} for name in body_positions}
table_body = {name:{'nt1':{}, 'control':{}} for name in body_positions}

# filter out all where they or their match has no body position
# stratify to have one match for each nt1
subset_body = ss.filter(lambda x: 'body' in x and x.body.sampleRate==4).stratify()

for pos in body_positions:
    for group in ['nt1', 'control']:
        
        subset = subset_body.filter(lambda x: x.group==group)
        subset.stratify() # make sure we have equal number in both sets
        
        values = [np.nanmean((p.get_signal('body', only_sleeptime=True)==cfg.mapping_body[pos])) for i,p in enumerate(subset)]

        values = [v for v in values if v]
        table_body[pos][group]['values'] = values


functions.calc_statistics(table_body)
plotting.distplot_table(table_body, 'Distribution of body positions', xlabel='ration spent in this position', ylabel='epochs spent in this position')

#########################
#%% Body position per sleep stage changes
#########################

body_positions = [cfg.mapping_body[x] for x in range(1,7)]
body_positions.remove('upside down') # silly you, sleeping like a bat

table_body = {stage:{name:{'nt1':{}, 'control':{}} for name in body_positions} for stage in range(5)}


for stage in tqdm(range(5), desc=f'Body position'):
    for pos in body_positions:
        for group in ['nt1', 'control']:
            subset = ss.filter(lambda x: x.group==group and 'body' in x and x.body.sampleRate==4)
            p=subset[0]
            
            values = [np.nanmean(p.get_signal('body', only_sleeptime=True, stage=stage)==cfg.mapping_body[pos]) for p in subset]
            # values = [v for v in values if v>0]
            table_body[stage][pos][group]['values'] = values


    functions.calc_statistics(table_body[stage])
    plotting.distplot_table(table_body[stage], f'Body positions in stage {cfg.num2stage[stage]}',
                                      xlabel='Ratio of this position in this stage',
                                      ylabel='Number of patients with this ratio')
 

    
plotting.print_table_with_subvars(table_body, f'Body positions in stages')

#%% nr of position changes

features = ['Total position changes']
table_body_changes = {name:{'nt1':{}, 'control':{}} for name in features}

for name in features:
    for group in ['nt1', 'control']:
        subset = ss.filter(lambda x: x.group==group and 'body' in x and x.body.sampleRate==4)
        values = [np.count_nonzero(np.diff(p.get_signal('body', only_sleeptime=True))) for p in subset]
        table_body_changes[name][group]['values'] = values
        
functions.calc_statistics(table_body_changes)
plotting.print_table(table_body_changes, f'Total body position changes')
plotting.distplot_table(table_body_changes, f'Total body position changes', columns=1)

#%%
stimer.stop('All calculations')

