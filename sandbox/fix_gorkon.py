# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:40:36 2020

add

@author: skjerns
"""
import config
import os
from tqdm import tqdm
import ospath
from pyedflib import highlevel


folder = "Z:/NT1-HRV-data"
new_folder = ospath.join(folder, "new")
os.makedirs(new_folder, exist_ok=True)
mapping = config.channel_mapping


files = ospath.list_files(folder, exts='edf')
for file in tqdm(files):
    name = ospath.basename(file)[:-4]
    new_file = ospath.join(new_folder, name + ".edf")
    if os.path.exists(new_file):
        print(f"{new_file} exists, skipping")
        continue
    
    highlevel.anonymize_edf(file, new_file, to_remove = ['patientcode', 'patientname'],
                            new_values  = [name, name], verify=False)
    highlevel.rename_channels(new_file, mapping=mapping, new_file=new_file)