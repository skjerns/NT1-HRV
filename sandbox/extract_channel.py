# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:43:58 2020

This file will extract the ECG channel from EDF files,
rename it to EGC I, to be compliant with Kubios

@author: skjerns
"""
import os
import ospath
import config as cfg
import sleep_utils
from pyedflib import highlevel
from tqdm import tqdm



copy_folder = 'z:/ecg/'
edf_file = 'Z:/mnc/dhc/training/N0020-nsrr.edf'
def extract_ecg(edf_file, copy_folder):
    filename = os.path.basename(edf_file)
    new_edf_file = os.path.join(copy_folder, filename)
    if os.path.exists(new_edf_file): return
    try:
        header = highlevel.read_edf_header(edf_file)
    except:
        print(f'error in file {edf_file}')
        return
    channels = header['channels']
    try:
        channels.remove('cs_ECG')
    except:
        print(f'warning, {edf_file} has no cs_ECG')
    ch_names = [x for x in channels if 'ECG' in x.upper()]
    if len(ch_names)>1:
        print(f'Warning, these are present: {ch_names}, selecting {ch_names[0]}')
    ch_name = ch_names[0]

    signals, shead, header = highlevel.read_edf(edf_file, ch_names=[ch_name], digital=True, verbose=False)

    shead[0]['label'] = 'ECG'


    assert len(signals)>0, 'signal empty'
    try:
        highlevel.write_edf(new_edf_file, signals, shead, header, digital=True)
    except:
        shead[0]['digital_min'] = signals.min()
        shead[0]['digital_max'] = signals.max()
        highlevel.write_edf(new_edf_file, signals, shead, header, digital=True)

if __name__ == '__main__':
    ecg = cfg.ecg_channel
    mnc_folder = cfg.folder_mnc
    files = ospath.list_files(mnc_folder, subfolders=True, exts=['.edf'])

    for edf_file in tqdm(to_extract):
        extract_ecg(edf_file, copy_folder)