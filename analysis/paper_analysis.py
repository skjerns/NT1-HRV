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
import stimer
import functions
import features
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
# ss = ss.filter(lambda x: x.drug_hrv==0 and x.drug_sleep==0)
ss = ss.stratify() # only use matched participants
p = ss[2]

# apply this type of pvalue correction per table,
# see  from statsmodels.stats.multitest.multipletests
correction = 'holm'

# only analyse the first 3 REM cycles
max_epochs = int(2*60*4.5)

#%% Population description
descriptors = ['gender', 'age', 'Number of epochs (only sleeptime)', 'Artefact ratio']
table_general = {name:{'nt1':{}, 'control':{}} for name in descriptors}

total_number_of_epochs =  sum([len(p.get_hypno(only_sleeptime=True)) for p in ss])
total_number_discarded = sum([sum(p.get_artefacts(only_sleeptime=True)) for p in ss])

print(f'{total_number_of_epochs} are available, of which {total_number_discarded} are discarded ({total_number_discarded/total_number_of_epochs*100:.1f}%)')

for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)

    # gender
    table_general['gender'][group]['female'] = np.sum([x.gender=='female' for x in subset])
    table_general['gender'][group]['male'] = np.sum([x.gender=='male' for x in subset])
    table_general['gender'][group]['values'] = [x.gender=='male' for x in subset]
    table_general['gender'][group]['mean'] = 5
    table_general['gender'][group]['std'] = None
    table_general['gender'][group]['p'] = 2

    # number of epochs
    values = list([len(p.get_hypno(only_sleeptime=True)) for p in subset])
    table_general['Number of epochs (only sleeptime)'][group]['values'] = values

    # number of discarded epochs
    values = list([np.nanmean(p.get_artefacts(only_sleeptime=True, wsize=300)) for p in subset])
    table_general['Artefact ratio'][group]['values'] = values

    # age
    values = np.array([x.age for x in subset])
    table_general['age'][group]['values'] = values


# calculate mean, std and p values for the values above
functions.calc_statistics(table_general)
# plot distributions for the two groups and save results to html
plotting.print_table(table_general, '1 Population description', correction=correction)
table_general.pop('gender') # remove this, can't be plotted appropriately
plotting.distplot_table(table_general, '1 Population description', ylabel='counts')

# now print some general channel information
ss.summary()

#%%### Sleep Stage Parameters

descriptors = ['TST', 'sleep efficiency', 'S1 latency' ,
               'S2 latency','SWS latency','REM latency',
               'S1 ratio', 'S2 ratio', 'SWS ratio', 'REM ratio', 'WASO ratio',
               'Stage shift index', 'Arousals', 'Arousal transitions',
               'Nr. of Awakenings', 'Awakening Lengths', #'Awakenings/Hour'
               ]

table1 = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)

    # TST in minutes
    values = np.array([len([x for x in p.get_hypno(only_sleeptime=True)[:max_epochs] if x in [1,2,3,4]])/2 for p in subset])
    table1['TST'][group]['values'] = values
    
    # Sleep efficiency
    values = np.array([len(p.get_hypno(only_sleeptime=True)[:max_epochs]) for p in subset])
    values =  table1['TST'][group]['values'] / values*2
    table1['sleep efficiency'][group]['values'] = values
   
    # S1 latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)[:max_epochs]==1)/2 for p in subset])
    table1['S1 latency'][group]['values'] = values
   
    # S2 latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)[:max_epochs]==2)/2 for p in subset])
    table1['S2 latency'][group]['values'] = values
    
    # SWS latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)[:max_epochs]==3)/2 for p in subset])
    table1['SWS latency'][group]['values'] = values
    
    # REM latency in minutes
    values = np.array([np.argmax(p.get_hypno(only_sleeptime=True)[:max_epochs]==4)/2 for p in subset])
    table1['REM latency'][group]['values'] = values
    
    # Sleep stage 1 ratio
    values = np.array([np.nanmean(p.get_hypno(only_sleeptime=True)[:max_epochs]==1) for p in subset])
    table1['S1 ratio'][group]['values'] = values

     # Sleep stage 2 ratio
    values = np.array([np.nanmean(p.get_hypno(only_sleeptime=True)[:max_epochs]==2) for p in subset])
    table1['S2 ratio'][group]['values'] = values

     # SWS ratio
    values = np.array([np.nanmean(p.get_hypno(only_sleeptime=True)[:max_epochs]==3) for p in subset])
    table1['SWS ratio'][group]['values'] = values
    
     # REM ratio
    values = np.array([np.nanmean(p.get_hypno(only_sleeptime=True)[:max_epochs]==4) for p in subset])
    table1['REM ratio'][group]['values'] = values

    # WASO ratio
    values = np.array([np.nanmean(p.get_hypno(only_sleeptime=True)[:max_epochs]==0) for p in subset])
    table1['WASO ratio'][group]['values'] = values

    # Stage shift index, nr of stage changes per hour
    hypnos = [p.get_hypno(only_sleeptime=True)[:max_epochs] for p in subset]
    values = np.array([np.count_nonzero(np.diff(hypno)) for hypno in hypnos])
    values = values / (max_epochs/2/60)
    table1['Stage shift index'][group]['values'] = values
    
    # LM index (movements)
    # PLM index
    # we dont have these annotations, so skip
    
    # Arousal transitions
    hypnos = [p.get_hypno(only_sleeptime=True)[:max_epochs] for p in subset]
    arousals = [p.get_arousals(only_sleeptime=True) for p in subset]
    arousals = [v[:np.argmax(v>(max_epochs-1))] for v in arousals]
    values = [arousal_transitions(h, a) for h, a in zip(hypnos, arousals)]
    for h,a in zip(hypnos, arousals): arousal_transitions(h,a)
    table1['Arousal transitions'][group]['values'] = values
    
    #### Additional
    
    # Number of arousals
    arousals = [p.get_arousals(only_sleeptime=True) for p in subset]
    arousals = [v[:np.argmax(v>max_epochs-1)] for v in arousals]
    values = [len(v) for v in arousals]
    table1['Arousals'][group]['values'] = values

    # Number of awakenings
    values = list([len(functions.phase_lengths(h[:max_epochs])[0]) for h in subset.get_hypnos(only_sleeptime=True)])
    table1['Nr. of Awakenings'][group]['values'] = values
    
    # Number of awakenings//hour
    # REMOVED, as we take all values for 4.5 anyway
    #values = list([len(functions.phase_lengths(h[:max_epochs])[0])/(len(h[:max_epochs])/120) for h in subset.get_hypnos(only_sleeptime=True)])
    #table1['Awakenings/Hour'][group]['values'] = values
    
    # Length of awakenings
    values = list([np.nanmean(functions.phase_lengths(h[:max_epochs])[0])/2 for h in subset.get_hypnos(only_sleeptime=True)])
    table1['Awakening Lengths'][group]['values'] = values



# calculate mean, std and p values for the values above
functions.calc_statistics(table1)
# plot distributions for the two groups and save results to html
plotting.print_table(table1, '2 Sleep Stage Parameters', correction=correction)
plotting.distplot_table(table1, '2 Sleep Stage Parameters', ylabel='counts')

#%% Van Meijden Figure 2 Mean episode length  2015 ,
# stage_means = dict(zip(range(6), [{'control':{}, 'nt1':{}} for _ in range(6)]))
# fig, axs = plt.subplots(2, 2)

# for group in ['nt1', 'control']:
#     subset = ss.filter(lambda x: x.group==group and hasattr(x, 'feats.pkl'))
#     hypnos = subset.get_hypnos(only_sleeptime=True)
#     phase_counts = [functions.phase_lengths(h[:max_epochs]) for h in hypnos]
#     phase_counts_all = dict(zip(range(6), [[] for _ in range(6)]))
#     for phase_count in phase_counts: 
#         for i in phase_count:
#             phase_counts_all[i].extend(phase_count[i])
            
    
#     for stage in range(6):
#         durations = np.array(phase_counts_all[stage], dtype=int)/2
#         stage_means[stage][group]['values'] = durations
#         if stage in [0, 5]: continue
    
#         hist = np.histogram(durations, bins=np.arange(16))
#         ax = axs.flatten()[stage-1]
#         ax.plot(hist[1][:-1], hist[0])
#         ax.set_title(cfg.num2stage[stage])
#         ax.set_xlabel('minutes')
#         ax.set_ylabel('total # of epochs')
#         ax.legend(['nt1', 'control'])

# n = len(ss.filter(lambda x: hasattr(x, 'feats.pkl') and not x.ecg_broken=='False'))
# plt.suptitle(f'Number of sleep phases for different phase lengths, n={n}')

# # calculate statistics and print out results
# functions.calc_statistics(stage_means)
# plotting.print_table(stage_means, 'Number of sleep phases for different phase lengths', correction=correction)

# for i, stage in enumerate([1,2,3,4]):
#     ax = axs.flatten()[i]
#     minmean = min(stage_means[stage]['nt1']['mean'], stage_means[stage]['control']['mean'])
#     ax.axvline(minmean, linestyle='dashed', linewidth=2, c='black')
    
#%% Van Meijden Figure 3: Feat Change after transition

feat_names = ['mean_HR', 'LF_power', 'HF_power', 'LF_HF']
post_transitions = {feat:{stage:{} for stage in range(6)} for feat in feat_names}
   
for name in feat_names:
    for stage in tqdm(range(6), desc=f'feat {name}'):
        for group in ['nt1', 'control']:
            subset = ss.stratify(lambda x: hasattr(x, 'feats.pkl') and x.ecg_broken==False).filter(lambda x: x.group==group)
            values = []
            for p in subset:
                hypno = p.get_hypno(only_sleeptime=True)[:max_epochs]
                # get the phase sequence, the lengths of the phase,
                # and their start in the hypnogram
                stages, lengths, idxs = functions.sleep_phase_sequence(hypno)
                # get all indices where we have the given stage with longer
                # period than the mean stage length
                # minlen = min(stage_means[stage]['nt1']['mean'], stage_means[stage]['control']['mean'])
                minlen = 10#int(minlen)
                # we need it in epoch notation, so *2
                phase_starts = np.where(np.logical_and(stages==stage, lengths>=minlen))[0]
                feat = p.get_feat(name, only_sleeptime=True, wsize=300, step=30)[:max_epochs]

                # loop through all phases that are longer than minlen and copy features from there
                for start in phase_starts:
                    feat_values = feat[idxs[start]: idxs[start]+minlen]
                    # if there is a NAN value, skip this epoch.
                    if np.isnan(feat_values).any(): continue
                    values.append(feat_values[:minlen]) # only take %minmean% epochs

            post_transitions[name][stage][group] = {'values':np.atleast_2d(values)}
            
    # calculate statistics and print out results
    title = f'3 after transition, {name} rate changes directly after Sleep Phase Change'
    functions.calc_statistics(post_transitions[name])
    n = len(ss.filter(lambda x: hasattr(x, 'feats.pkl')))
    fig, axs = plotting.lineplot_table(post_transitions[name], title, xlabel='epochs', 
                                       ylabel=name, n=n)
    plotting.print_table(post_transitions[name], title, correction=correction)
    # break

#%% Van Meijden Table 2



# ????? I don't understand the mixed effect regression model they are used
#       or how to reproduce it...



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
    values = list([functions.starting_sequence_pizza(h[:max_epochs]) for h in subset.get_hypnos(only_sleeptime=True)])
    pizza2015['Starting Sequences \nN1-N2-SWS-REM / N2-REM / SOREM / Other'][group]['values'] = values
  
    # Sleep Stage Sequences (own definition, see functions.py)
    values = list([functions.starting_sequence(h[:max_epochs]) for h in subset.get_hypnos(only_sleeptime=True)])
    pizza2015['Starting Sequences \nSWS-REM / N2-REM / SOREM'][group]['values'] = values
    
    # Sleep Stage Sequences tW-Si (see Pizza 2015)
    values = np.array(list([functions.transition_index(h[:max_epochs], [0]) for h in subset.get_hypnos()]))
    values = values / table1['TST'][group]['values']*60
    pizza2015['Sequences W-S'][group]['values'] = values
    
    # Sleep Stage Sequences tW-NR-Ri (see Pizza 2015)
    values = np.array(list([functions.transition_index(h[:max_epochs], [0,1,4]) for h in subset.get_hypnos()]))
    values += np.array(list([functions.transition_index(h[:max_epochs], [0,2,4]) for h in subset.get_hypnos()]))
    values = values / table1['TST'][group]['values']*60
    pizza2015['Sequences W-NR-R'][group]['values'] = values
    
    # Sleep Stage Sequences N1-NR-Ri (see Pizza 2015)
    values = np.array(list([functions.transition_index(h[:max_epochs], [0,1,2,4]) for h in subset.get_hypnos()]))
    values += np.array(list([functions.transition_index(h[:max_epochs], [0,1,3,4]) for h in subset.get_hypnos()]))
    values = values / table1['TST'][group]['values']*60
    pizza2015['Sequences N1-NR-R'][group]['values'] = values
    
# calculate mean, std and p values for the values above
functions.calc_statistics(pizza2015)

# plot distributions for the two groups and print out table
plotting.print_table(pizza2015, '4.1 Sleep transition indices' , correction=correction)
plotting.distplot_table(pizza2015, '4.1 Sleep transition indices', columns=2,
                        xlabel=[*['Nr of Starting Sequence']*2, *3*['Nr of occurence / h']])


#%% Transition possibilities and compare between groups
descriptors = list(permutations([0,1,2,4], 2)) # all state transitions that are possible
descriptors.extend([(2,3), (3,2)])

transitions = {name:{'nt1':{}, 'control':{}} for name in descriptors}
for group in ['nt1', 'control']:
    subset = ss.filter(lambda x: x.group==group)
    for pair in descriptors:
        values = [len(functions.search_sequences(p.get_hypno()[:max_epochs], np.array(pair))) for p in subset]
        nr_transitions = [np.count_nonzero(np.diff(p.get_hypno(only_sleeptime=True)[:max_epochs])) for p in subset]
        transitions[pair][group]['values'] = np.array(values)/nr_transitions
        
        # comment this in for transitions / hour
        # tst = [len(p.get_hypno(only_sleeptime=True))/120 for p in subset]
        # transitions[pair][group]['values'] = np.array(values)/np.array(tst)
       
        # comment this in for total number of transitions
        # transitions[pair][group]['values'] = np.array(values)/np.array(tst)
         
       
        
# calculate mean, std and p values for the values above
functions.calc_statistics(transitions)
plotting.distplot_table(transitions, '4.2 % of transitions of total transitions', columns=3,
                        xlabel='4.2 % of all transitions', ylabel='count')
plotting.print_table(transitions, '% of transitions of total transitions', correction=correction )



#########################
#%% Body position changes
#########################

# body_positions = [cfg.mapping_body[x] for x in range(1,7)]
# table_body = {name:{stage:{'nt1':{}, 'control':{}} for stage in range(5)} for name in body_positions}
# table_body = {name:{'nt1':{}, 'control':{}} for name in body_positions}

# # filter out all where they or their match has no body position
# # stratify to have one match for each nt1
# subset_body = ss.filter(lambda x: 'body' in x and x.body.sampleRate==4).stratify()

# for pos in body_positions:
#     for group in ['nt1', 'control']:
        
#         subset = subset_body.filter(lambda x: x.group==group)
#         subset.stratify() # make sure we have equal number in both sets
        
#         values = [np.nanmean((p.get_signal('body', only_sleeptime=True)==cfg.mapping_body[pos])) for i,p in enumerate(subset)]

#         values = [v for v in values if v]
#         table_body[pos][group]['values'] = values


# functions.calc_statistics(table_body)
# plotting.distplot_table(table_body, '5.1 Distribution of body positions', xlabel='ration spent in this position', ylabel='epochs spent in this position')

# #########################
# #%% Body position per sleep stage changes
# #########################

# body_positions = [cfg.mapping_body[x] for x in range(1,7)]
# body_positions.remove('upside down') # silly you, sleeping like a bat

# table_body = {stage:{name:{'nt1':{}, 'control':{}} for name in body_positions} for stage in range(5)}


# for stage in tqdm(range(5), desc=f'Body position'):
#     for pos in body_positions:
#         for group in ['nt1', 'control']:
#             subset = ss.filter(lambda x: x.group==group and 'body' in x and x.body.sampleRate==4)
#             p=subset[0]
            
#             values = [np.nanmean(p.get_signal('body', only_sleeptime=True, stage=stage)==cfg.mapping_body[pos]) for p in subset]
#             # values = [v for v in values if v>0]
#             table_body[stage][pos][group]['values'] = values


#     functions.calc_statistics(table_body[stage])
#     plotting.distplot_table(table_body[stage], f'Body positions in stage {cfg.num2stage[stage]}',
#                                       xlabel='Ratio of this position in this stage',
#                                       ylabel='Number of patients with this ratio')
 

    
# plotting.print_table_with_subvars(table_body, f'5.2 Body positions in stages', correction=correction)

# #%% nr of position changes

# features = ['Total position changes']
# table_body_changes = {name:{'nt1':{}, 'control':{}} for name in features}

# for name in features:
#     for group in ['nt1', 'control']:
#         subset = ss.filter(lambda x: x.group==group and 'body' in x and x.body.sampleRate==4)
#         values = [np.count_nonzero(np.diff(p.get_signal('body', only_sleeptime=True))) for p in subset]
#         table_body_changes[name][group]['values'] = values
        
# functions.calc_statistics(table_body_changes)
# plotting.print_table(table_body_changes, f'5.3 Total body position changes', correction=correction)
# plotting.distplot_table(table_body_changes, f'5.3 Total body position changes', columns=1)



#%% Features in sleep stages: HF/LF, HRV, etc

features = ['mean_HR', 'VLF_power', 'LF_power', 'HF_power', 'LF_HF', 'RMSSD',
            'pNN50', 'SDNN']
table_features = {name:{stage:{'nt1':{}, 'control':{}} for stage in range(5)} for name in features}
masks = {'nt1':{}, 'control':{}}

for group in ['nt1', 'control']:
    subset = ss.stratify(lambda x: hasattr(x, 'feats.pkl') and x.ecg_broken==False).filter(lambda x: x.group==group)

    phase_counts = [functions.phase_lengths(h[:max_epochs]) for h in subset.get_hypnos(only_sleeptime=True)]
    phase_lengths = [functions.stage2length(h[:max_epochs]) for h in subset.get_hypnos(only_sleeptime=True)]

    for stage in range(5):
        
        ##### here we filter out which epochs are elegible for analysis
        # only take epochs that are longer than the mean epoch length
        # minlen = min(stage_means[stage]['nt1']['mean'], stage_means[stage]['control']['mean'])
        minlen = 2 # 5 minutes
        mask_length = [np.array(l)>=minlen for l in phase_lengths]
        # onlt take epochs of the given stage
        mask_stage = [h[:max_epochs]==stage for h in subset.get_hypnos(only_sleeptime=True)]
        # only take epochs with no artefacts surrounding 300 seconds
        mask_art = [p.get_artefacts(only_sleeptime=True, wsize=300)[:max_epochs]==False for p in subset]
        assert all([len(x)==len(y) for x,y in zip(mask_length, mask_stage)])
        assert all([len(x)==len(y)  for x,y in zip(mask_length, mask_art)])
        # create a singel mask
        mask = [np.logical_and.reduce((x,y,z)) for x,y,z in zip(mask_length, mask_stage, mask_art)]
        masks[group][stage] = mask    


for feat in tqdm(features, desc='Calculating features'):
    for stage in range(5):
        for group in 'nt1', 'control':
            subset = ss.stratify(lambda x: hasattr(x, 'feats.pkl') and x.ecg_broken==False).filter(lambda x: x.group==group)
            # get all values
            feat_name = cfg.mapping_feats[feat]
            values = [p.get_feat(feat_name, only_sleeptime=True)[:max_epochs] for p in subset]
            # combine values with masks
            values = [np.nanmean(v[m[:len(v)]]) for v, m in zip(values, masks[group][stage]) if len(v[m[:len(v)]])!=0]
            table_features[feat][stage][group] = {'values':values}

    # manually calculate WBSO (wake before sleep onset)
    table_features[feat]['WAKE Before Sleep'] = {}
    for group in ['nt1', 'control']:
        subset = ss.stratify(lambda x: hasattr(x, 'feats.pkl') and x.ecg_broken==False).filter(lambda x: x.group==group)
        sleep_onset = [np.argmax((hyp==1) | (hyp==2) | (hyp==3) | (hyp==4)) for hyp in subset.get_hypnos(only_sleeptime=False)]
        values = [np.nanmean(p.get_feat(feat_name, only_sleeptime=False)[minlen*2:m]) for p, m in zip(subset, sleep_onset)]

        table_features[feat]['WAKE Before Sleep'][group] = {'values':values}


    functions.calc_statistics(table_features[feat])
    plotting.distplot_table(table_features[feat], title=f'9 Feature {feat} during sleep stages')
    
plotting.print_table_with_subvars(table_features, title=f'9 Features during during sleep stages', correction=correction)

#%% ULF analysis
feat_names = ['ULF']
table_whole_night = {feat:{'nt1':{}, 'control':{}} for feat in feat_names}

for group in ['nt1', 'control']:
    subset = ss.stratify(lambda x: hasattr(x, 'feats.pkl') and x.ecg_broken==False).filter(lambda x: x.group==group)
    RRs_tuples = [p.get_RR(only_sleeptime=True) for p in subset]
    RRs = [RR[:np.argmax(T_RR>=3600*2+T_RR[0])] for T_RR, RR in RRs_tuples]
    ulf = [features.ULF_power(RR) for RR in tqdm(RRs, desc='ULF calc')]

    table_whole_night['ULF'][group]['values'] = np.array(ulf)

functions.calc_statistics(table_whole_night)
plotting.distplot_table(table_whole_night, 'ULF')


stimer.stop('All calculations')

