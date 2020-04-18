# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:33:49 2020

@author: skjerns
"""
from itertools import groupby
import numpy as np

def phase_lengths(hypno):
    """calculate the length of each sleep phase"""    
    lengths = [sum(1 for i in g) for k,g in groupby(hypno)]
    stages = [hypno[sum(lengths[:i+1])-1] for i in range(len(lengths))]
    
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