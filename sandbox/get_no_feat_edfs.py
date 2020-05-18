# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:33:03 2020

@author: skjerns
"""
import ospath
import os, sys
import numpy as np
import shutil
import config as cfg
from tqdm import tqdm
from sleep import SleepSet
from pyedflib.highlevel import drop_channels


files = ospath.list_files(cfg.folder_edf, exts='edf')
mats = ospath.list_files(cfg.folder_edf, exts='mat', relative=True)
mats = [mat.replace('_hrv', '').replace('_small', '').replace('.mat', '') for mat in mats]

ecg_dir = cfg.folder_edf + '/edf_no_mat'
os.makedirs(ecg_dir)


for file in tqdm(files):
    code = ospath.basename(file)[:-4]
    if code in mats: continue
    drop_channels(os.path.join(cfg.folder_edf, f'{code}.edf'), os.path.join(ecg_dir,f'{code}.edf'), to_keep=['ECG I'])