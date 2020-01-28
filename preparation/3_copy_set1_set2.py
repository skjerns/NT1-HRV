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
import shutil
import ospath
import config as cfg
from tqdm import tqdm

def load_csv(csv_file, sep=';'):
    with open(csv_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        lines = [line for line in lines if not line.startswith('#')]
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line!='']
        lines = [line.split(';') for line in lines]
        return lines

if __name__=='__main__':
    documents = cfg.documents
    datasets = [ospath.join(documents, 'mapping_' + d + '.csv') for d in cfg.datasets]
    matching = cfg.matching
    set1_path = ospath.join(cfg.data, 'set1')
    set2_path = ospath.join(cfg.data, 'set2')
    
    
    matchings = load_csv(matching)
    
    set1 = load_csv(datasets[0])
    set2 = load_csv(datasets[1])
    
    os.makedirs(ospath.join(cfg.data, 'set1'), exist_ok=True)
    os.makedirs(ospath.join(cfg.data, 'set2'), exist_ok=True)
    os.makedirs(ospath.join(cfg.data, 'set1', 'not_matched'), exist_ok=True)
    os.makedirs(ospath.join(cfg.data, 'set2', 'not_matched'), exist_ok=True)
    # copy the files into set1 and set2 respectively
    
    for p_orig, p_coded, _, c_coded, diff in tqdm(matchings):
        if int(diff)>cfg.max_age_diff:break
        for patient, p_coded1 in set1:
            if patient==p_orig:
                assert p_coded==p_coded1 # sanity check
                old_location = ospath.join(cfg.data, p_coded +'.edf')
                new_location = ospath.join(cfg.data, 'set1', p_coded +'.edf')
                if not ospath.exists(new_location):
                    shutil.copy(old_location, new_location)
                    
        for patient, p_coded1 in set2:
            if patient==p_orig:
                assert p_coded==p_coded1 # sanity check
                old_location = ospath.join(cfg.data, p_coded +'.edf')
                new_location = ospath.join(cfg.data, 'set2', p_coded +'.edf')
                if not ospath.exists(new_location):
                    shutil.copy(old_location, new_location)

    # now we also copy the non-matched files
                    
    for p_orig, p_coded in tqdm(set1):
        matched = [match[0] for match in matchings]
        if p_orig not in matched:
            old_location = ospath.join(cfg.data, p_coded +'.edf')
            new_location = ospath.join(cfg.data, 'set1', 'not_matched', p_coded +'.edf')
            if not ospath.exists(new_location):
                shutil.copy(old_location, new_location)
                    
    for p_orig, p_coded in tqdm(set2):
        matched = [match[0] for match in matchings]
        if p_orig not in matched:
            old_location = ospath.join(cfg.data, p_coded +'.edf')
            new_location = ospath.join(cfg.data, 'set2', 'not_matched', p_coded +'.edf')
            if not ospath.exists(new_location):
                shutil.copy(old_location, new_location)
