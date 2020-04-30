# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:57:00 2020

@author: nd269
"""
import config as cfg
import numpy as np
import scipy
from scipy import stats

# create empty dummy functions
_locals = locals()
for key, name in cfg.mapping_feats.items():
    _locals[name] = lambda *args, **kwargs:  False

def dummy(ecg, **kwargs):
    """
    each function here should be named exactly as the feature 
    name in config.features_mapping.
    Like this the feature extraction can be done automatically.
    The function can accept the following parameters:
        
    ecg, rr, kubios
    
    all functions should accept **kwargs that are ignored.
    """
# 1 
def mean_HR(kubios, **kwargs):
    data = kubios['TimeVar']['mean_HR']
    return data.squeeze()

# 2
def mean_RR(kubios, **kwargs):
    data = kubios['TimeVar']['mean_RR']
    return data.squeeze()

# 4
def RMSSD(kubios, **kwargs):
    data = kubios['TimeVar']['RMSSD']
    return data.squeeze()

# 6
def pNN50(kubios, **kwargs):
    data = kubios['TimeVar']['pNNxx']
    return data.squeeze()

# 10
def LF(kubios, **kwargs):
    data = kubios['TimeVar']['LF_power']
    return data.squeeze()

# 11
def HF(kubios, **kwargs):
    data = kubios['TimeVar']['HF_power']
    return data.squeeze()

# 12
def LF_HF(kubios, **kwargs):
    data = kubios['TimeVar']['LF_power']/kubios['TimeVar']['HF_power']
    return data.squeeze()