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
import ospath
import unisens
import features
import numpy as np
import config as cfg
from unisens import Unisens, CustomEntry, EventEntry, ValuesEntry
from joblib import Parallel, delayed

#%%

def extract_features(folder):
    pass
#%%   
    u = Unisens(folder)
    kubios = u.feats.get_data()
    feats = u.feats
    
    ValuesEntry(id='0_dummy.csv', parent=feats)
    
    ecg = u.ECG.get_data()
    sfreq = u.ECG.sampleRate
    rr = u.rr.get_times()
    
    
    for nr, name in cfg.feats_mapping.items():
        id = f'{name}.csv'
        if id in u: u.remove_entry(id)
        data = features.__dict__[name](ecg=ecg, rr=rr, kubios=kubios, sfreq=sfreq)
        if data is False: continue
        data = list(zip(np.arange(0, len(data)), data))
        feat = ValuesEntry(id=id, parent=feats)
        feat.set_data(data, sampleRate=1/30)
        feat.seg_len = kubios['Param']['Length_Segment']
    u.save()
    
#%%
if __name__=='__main__':
    documents = cfg.documents
    unisens_folder = cfg.folder_unisens
    
    folders = ospath.list_folders(unisens_folder)
    folder = folders[1]
    # Parallel(n_jobs=8, verbose=10)(delayed(extract_features)(folder) for folder in folders) 
