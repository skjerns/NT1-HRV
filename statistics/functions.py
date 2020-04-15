# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:33:49 2020

@author: skjerns
"""
from itertools import groupby
import numpy as np

def epoch_lengths(hypno):
    """calculate the length of each sleep phase"""    
    
    
    lengths = [sum(1 for i in g) for k,g in groupby(hypno)]
    stages = [hypno[sum(lengths[:i+1])-1] for i in range(len(lengths))]
    
    histogram = dict(zip(np.arange(6), [[] for i in range(6)]))
    
    for s, l in zip(stages, lengths):
        histogram[s] += [l]
    return histogram