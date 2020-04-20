# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:43:58 2020

This file will extract a certain channel from all EDF files of a folder
and save it under '/extracted_CHANNEL_NAME'

@author: skjerns
"""
import os
import ospath
import config as cfg
import sleep_utils
from tqdm import tqdm

if __name__ == '__main__':
    ecg = cfg.ecg_channel
    data = cfg.folder_edf
    folders = ospath.list_folders(data, subfolders=True, add_parent=False)
    for folder in folders:
        if ecg in folder: continue
        files = ospath.list_files(folder, exts='edf')
        new_folder = ospath.join(folder, ecg)
        os.makedirs(new_folder, exist_ok=True)
        for file in tqdm(files):
            new_file = ospath.join(new_folder, ospath.basename(file))
            if ospath.exists(new_file): continue
            sleep_utils.drop_channels(file, new_file, to_keep=['ECG I'])
