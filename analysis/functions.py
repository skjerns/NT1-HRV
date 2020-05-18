# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:33:49 2020

@author: skjerns
"""
from itertools import groupby
import numpy as np
import scipy.stats as stats

significance_test = stats.mannwhitneyu
significance_test = stats.ttest_ind



def calc_statistics(table):
    """
    Calculates statistics such as mean, std and p value for a table
    
    a table is defined as follows
    
    dictionary['variable name']['group1/group2']['values'] = [0,5,2,3,4, ...]
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]

    """
    for descriptor in table:
        if 'nt1' in table[descriptor]: # only one level
            if len(table[descriptor]['nt1'])==0 or len(table[descriptor]['control'])==0\
               or len(table[descriptor]['nt1']['values'])==0 or len(table[descriptor]['control']['values'])==0:
                table[descriptor]['p'] = '-'
                table[descriptor]['mean'] = np.nan
                table[descriptor]['std'] = np.nan
                continue
            values_nt1 = table[descriptor]['nt1']['values']
            values_cnt = table[descriptor]['control']['values']
            try:
                table[descriptor]['p'] = stats.mannwhitneyu(values_nt1, values_cnt).pvalue
            except:
                table[descriptor]['p'] = '-'
                
            for group in ['nt1', 'control']:
                table[descriptor][group]['mean'] = np.nanmean(table[descriptor][group]['values'])
                table[descriptor][group]['std'] = np.nanstd(table[descriptor][group]['values'])
                
        else: # with subvariables / dividede by stages
            calc_statistics(table[descriptor])

def sleep_phase_sequence(hypno):
    """
    a phase is a continuous time of a specific sleep stage
    
    get sleep stage with lengths
    eg [0,0,0,0,1,1,1,1,2,2] will return
    stages  = [0,1,2]
    lengths = [4,4,2]
    idxs = [0,4,8] # start of the phase
    """
    lengths = np.array([sum(1 for i in g) for k,g in groupby(hypno)])
    stages = np.array([hypno[sum(lengths[:i+1])-1] for i in range(len(lengths))])
    idxs = np.pad(np.cumsum(lengths),[1,0])[:-1]
    return stages, lengths, idxs


def transition_index(hypno, transition_pattern):
    """
    count occurences of sleep stage transition patterns
    
    e,g, find number of transitions W-S1-W , disregarding of W/S1 length
    defined by Pizza 2015: doi.org/10.5665/sleep.4908
    """
    stage_sequence, lengths, idxs = sleep_phase_sequence(hypno)
    positions = search_sequences(stage_sequence, transition_pattern)
    return len(positions)


def starting_sequence_pizza(hypno):
    """
    
    return which sequence sleep stages occur in a hypnogram
    defined by Pizza 2015: doi.org/10.5665/sleep.4908
    
    0 : N1 - N2 - N3 - REM   or   N2 - N3 - REM
    1 : N1 - N2 - REM        or   N2 - REM - N1
    2 : W/N1 - REM           or   N2 - W - N1 - N4
    -1: other (should not happen)
    """
    hypno = hypno[hypno!=5] # remove artefacts
    
    # now create order of first appearance of sleep stage
    _, index = np.unique(hypno, return_index=True)
    sequence = []
    for i in sorted(index):
        sequence.append(hypno[i])
    
    # remove wake, we're not interested in that
    sequence.remove(0)
    
    # check which sequence we got
    if sequence[:4]==[1,2,3,4]:
        sequence_nr = 0
    elif sequence[:3]==[1,2,4]:
        sequence_nr = 1
    elif sequence[:2]==[1,4] or sequence[0]==[4]:
        sequence_nr = 2
    else:
        sequence_nr = 3
    return sequence_nr

def starting_sequence(hypno):
    """
    
    return which sequence sleep stages occur in a hypnogram
    this is our own definition
    
    
    
    0:  'normal transitions': First SWS then REM
    1:  'kind of normal': First N2, no SWS, then REM
    3:  SOREM: No N2, no SWS, directly REM
    -1: other should not happen, 
        all of them should fall into one of the categories above, 
        unless there is no REM at all
    """
    # idx of first S2, SWS and REM
    # add artificial -9 so that no argmax can be 0. 
    # if it's zero we know that it isn't found in the hypnogram at all
    # e.g. there is no REM,SWS or S2 at all
    hypno = np.array([-9] + [x for x in hypno])
    s2, sws, rem = np.argmax(hypno==2), np.argmax(hypno==3), np.argmax(hypno==4)
    
    if s2==0: s2=99999
    if sws==0: sws=99999
    if rem==0: rem=99999
    
    if sws<rem: sequence_nr = 0
    elif s2<rem: sequence_nr = 1
    elif rem<sws: sequence_nr=2
    else:
        sequence_nr=-1
        print(f'no rem?')
    return sequence_nr



def search_sequences(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """
    arr = np.array(arr)
    seq = np.array(seq)
    assert arr.ndim==seq.ndim and arr.ndim==1, 'can only look in array not matrix'
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    return np.where(M)[0]


def phase_lengths(hypno):
    """calculate the length of each sleep phase"""    
    stages, lengths, idxs = sleep_phase_sequence(hypno)

    histogram = dict(zip(np.arange(6), [[] for i in range(6)]))
    
    for s, l in zip(stages, lengths):
        histogram[s] += [l]
    return histogram


def stage2length(hypno):
    """calculate the length of each sleep phase and return the hypnogram
    with each epoch replaced by the length of its epoch.
    eg:        [0,0,0,1,1,2,2,2,2,2,2,2,5,5,3,3,3,3,3,3,3]
    turns into [3,3,3,2,2,7,7,7,7,7,7,7,2,2,7,7,7,7,7,7,7]
    
    This way we can easily get a bool mask for epochs longer than 10 epochs by
    [lengths>10]
    """    
    lengths = [sum(1 for i in g) for k,g in groupby(hypno)]
    hypno_lengths = []
    for l in lengths:
        hypno_lengths.extend([l]*l)
        
    assert len(hypno) == len(hypno_lengths)
    return hypno_lengths


def arousal_transitions(hypno, arousals):
    
    
    arousal_transitions = 0
    for pos in arousals:
        pre = hypno[pos]
        post = hypno[pos+1]
        if pre == 3 and post in [2,1]:
            arousal_transitions+=1
        elif pre==2 and post==1:
            arousal_transitions+=1
    return arousal_transitions