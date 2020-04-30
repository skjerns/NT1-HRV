# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:13:55 2020

a script that scans the dataset folders, loads the header
and checks if there are any channels that are not in our mapping,
e.g. another name for ECG that has not been added to the mappings dictionary
that is present in config.py

@author: skjerns
"""
import config as cfg
import os
import sleep_utils
import misc
import ospath
from tqdm import tqdm
from pyedflib.highlevel import read_edf_header

ch_mapping = cfg.mapping_channels

missing = set()
if __name__ == '__main__':
    files = []
    for dataset in cfg.datasets.values():
        files.extend(ospath.list_files(dataset, exts='edf', subfolders=True))
    
    ch_mapping = cfg.mapping_channels
    
    for file in tqdm(files):
        channels = read_edf_header(file)['channels']
        for ch in channels:
            if not ch in ch_mapping and not ch in ch_mapping.values(): 
                missing.add(ch)
                
                print(ch)

