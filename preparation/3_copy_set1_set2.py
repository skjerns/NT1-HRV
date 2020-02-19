# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:37:48 2020

We have two datasets of NT1 patients.
This script splits the anonymized files into dataset1 and dataset2.
It only copies files for which we have a match.
Eg at current state 27/01/2020 we have set1 with 28 and set2 with 30 patients

@author: skjerns
"""
import os
from misc import read_csv
import shutil
import ospath
import config as cfg
from tqdm import tqdm


if __name__=='__main__':
    documents = cfg.documents
    datasets = [ospath.join(documents, 'mapping_' + d + '.csv') for d in cfg.datasets]
    matching = cfg.matching
    set1_path = ospath.join(cfg.data, 'set1')
    set2_path = ospath.join(cfg.data, 'set2')
    
    
    matchings = read_csv(matching)
    
    set1 = read_csv(datasets[0])
    set2 = read_csv(datasets[1])
    
    os.makedirs(ospath.join(cfg.data, 'set1'), exist_ok=True)
    os.makedirs(ospath.join(cfg.data, 'set2'), exist_ok=True)
    os.makedirs(ospath.join(cfg.data, 'set1', 'not_matched'), exist_ok=True)
    os.makedirs(ospath.join(cfg.data, 'set2', 'not_matched'), exist_ok=True)
    
    
    # copy the files into nt1:matched set1 and nt1:matched set2 respectively
    for p_orig, p_coded,gender, age, c_name, c_coded,c_gender, c_age, diff in tqdm(matchings):
        if int(diff)>cfg.max_age_diff:break
        for patient, p_coded1 in set1:
            if patient==p_orig:
                assert p_coded==p_coded1 # sanity check
                old_location_nt1 = ospath.join(cfg.data, p_coded +'.edf')
                new_location_nt1  = ospath.join(cfg.data, 'set1', p_coded +'.edf')
                if not ospath.exists(new_location_nt1):
                    shutil.copy(old_location_nt1, new_location_nt1)
                    
                old_location_cnt = ospath.join(cfg.data, c_coded +'.edf')
                new_location_cnt  = ospath.join(cfg.data.strip(), 'set1', c_coded.strip() +'.edf')
                if not ospath.exists(new_location_cnt):
                    shutil.copy(old_location_cnt, new_location_cnt)
                    
        for patient, p_coded1 in set2:
            if patient==p_orig:
                assert p_coded==p_coded1 # sanity check
                old_location_nt1 = ospath.join(cfg.data, p_coded +'.edf')
                new_location_nt1  = ospath.join(cfg.data, 'set2', p_coded +'.edf')
                if not ospath.exists(new_location_nt1):
                    shutil.copy(old_location_nt1, new_location_nt1)
                    
                old_location_cnt = ospath.join(cfg.data, c_coded +'.edf')
                new_location_cnt  = ospath.join(cfg.data, 'set2', c_coded +'.edf')
                if not ospath.exists(new_location_cnt):
                    shutil.copy(old_location_cnt, new_location_cnt)
                    
    # now we also copy the non-matched nt1 files (not the controls)
    matched = [match[1] for match in matchings if int(match[-1])<=cfg.max_age_diff]
    for p_orig, p_coded in tqdm(set1):
        if p_coded not in matched:
            old_location = ospath.join(cfg.data, p_coded +'.edf')
            new_location = ospath.join(cfg.data, 'set1', 'not_matched', p_coded +'.edf')
            if not ospath.exists(new_location):
                shutil.copy(old_location, new_location)
                    
    for p_orig, p_coded in tqdm(set2):
        if p_coded not in matched:
            old_location = ospath.join(cfg.data, p_coded +'.edf')
            new_location = ospath.join(cfg.data, 'set2', 'not_matched', p_coded +'.edf')
            if not ospath.exists(new_location):
                shutil.copy(old_location, new_location)
