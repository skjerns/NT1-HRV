# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:44:21 2020

Trying to classify sleep stages with HRV features.
Let's see if this works!

@author: Simon Kern
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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
import features


plt.close('all')
ss = SleepSet(cfg.folder_unisens)
ss = ss.stratify() # only use matched participants
p = ss[1]

#%% step 1: get features
if __name__=='__main__':
    wsize = 300
    step = 30
    
    feats = []
    hypnos = []

    for p in ss:
        p.reset()
        feats_p = []
        for feat_name in cfg.mapping_feats:
            if feat_name in features.__dict__:
                feat = p.get_feat(feat_name, wsize=wsize, step=step)
                feats_p.append(feat)
                hypno = p.get_hypno()
                hypnos.append(hypno)
        break