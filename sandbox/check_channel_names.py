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
import sleep_utils
import ospath
from tqdm import tqdm
remove = ['C3', 'C4', 'EOGl', 'EOGr', 'M1', 'M2', 'A1', 'A2', 'EOGV', 'Fz', 'Oz',
          'Plethysmogram', 'Pleth', 'Akku', 'Numeric Aux10', 'O1', 'O2', 'FlowPAP', 
          'Druck', 'li Arm', 'Skin potential', 'Numeric Aux9']
ch_mapping = cfg.channel_mapping


if __name__ == '__main__':
    files = []
    for dataset in cfg.datasets.values():
        files.append(ospath.list_files(dataset, exts='edf', subfolders=True))
    done = []
    missing = []
    for file in tqdm(files):
        print(file)
        if 'A9971.edf'in file: continue
        if ospath.basename(file) in done: continue
        try:
            header = sleep_utils.read_edf_header(file)
        except:
            pass
        chs = header['channels']
        for ch in chs:
            if ch in remove: 
                continue
            elif ch in ch_mapping:
                continue
            elif ch in ch_mapping.values():
                continue
            else:
                missing.append(ch)
                print('not in mapping:', ch)
        done.append(ospath.basename(file))

    
    
    
    
    
    