# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:37:48 2020

This script will load the previously created unisens objects
and extract the features from the features.pkl which has the
data created by kubios.

As not all features are implemented by kubios, additional
features are extracted using functions defined in features.py.

@author: skjerns
"""
import os
import stimer
import ospath
import unisens
import features
import numpy as np
import config as cfg
from tqdm import tqdm
from sleep import Patient, SleepSet
from unisens import Unisens, CustomEntry, EventEntry, ValuesEntry
from joblib import Parallel, delayed

#%%

def extract_features(patient):
    pass
#%%   
    kubios = patient.feats.get_data()
    feats = patient.feats
    
    ecg = patient.ECG.get_data()
    sfreq = patient.ECG.sampleRate
    rr = patient.rr.get_times()
    
    
    for nr, name in cfg.feats_mapping.items():
        id = f'feats/{name}.csv'
        if id in patient: patient.remove_entry(id)
        data = features.__dict__[name](ecg=ecg, rr=rr, kubios=kubios, sfreq=sfreq)
        if data is False: continue
        data = list(zip(np.arange(0, len(data)), data))
        feat = ValuesEntry(id=id, parent=feats._folder)
        feat.set_data(data, sampleRate=1/30)
        feat.seg_len = kubios['Param']['Length_Segment']
        feats.add_entry(feat, stack=False)
    patient.save()
    
#%%
if __name__=='__main__':
    documents = cfg.documents
    unisens_folder = cfg.folder_unisens
    miss = []
    ss = SleepSet(unisens_folder, readonly=False)
    patient = ss[1]
    for patient in tqdm(ss):
        if not 'feats' in patient:
            miss.append(patient.code)
            continue
        extract_features(patient)
    if len(miss)>0: print(f'the following have no matfile/feats.pkl: {miss}')
