# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:37:48 2020

This script will turn the codified EDFs into a folderstructure in the
Unisens format using pyUnisens.

The following information will be converted / retained, and added:
    
    1. Main EEG channels, EKG channel + information as SignalEntry
    2. Hypnogram (if present) as ValuesEntry
    3. The kubios main (mat) file as CustomEntry
    4. EDF annotations as EventEntry
    
These meta attributes will be added:
    - recording length / epoch length
    - recording onset
    - sampling frequency of ECG
    - total number of discarded episodes (manual annotation and kubios)
    - group this edf belongs to
    - 

@author: skjerns
"""
import shutil
import mat73
import pyedflib
import unisens
from unisens import SignalEntry, Unisens
import os
from misc import read_csv
import shutil
import ospath
import sleep_utils
import config as cfg
from tqdm import tqdm



def to_unisens(edf_file, delete=False):
    name = ospath.basename(edf_file)[:-4]
    folder = ospath.dirname(edf_file)
    
    # get all additional files that belong to this EDF
    add_files = ospath.list_files(folder, patterns=name + '*')
    u = Unisens(ospath.join(folder, name))
    signal, shead, header = pyedflib.highlevel.read_edf(edf_file, ch_names='ECG I')
    annotations = header['annotations']
    
    u.add_entry(SignalEntry(id='ECG', parent=u).set_data(signal.astype(np.float32)))
    u.sampling_frequency = shead[0]['sample_rate']
    u.length = signal.shape[1]//int(u.sampling_frequency)
    u.epochs = signal.shape[1]//int(u.sampling_frequency)//30
    
    for file in add_files:
        if file.endswith('txt') or file.endswith('dat'):
            hypno = sleep_utils.read_hypnogram(file)
        elif file.endswith('mat'):
            mat = mat73.loadmat(file)
        elif file.endswith('npy'):
            art = np.load(file)
        elif file.endswith('.edf'):
            pass
        else:
            raise Exception(f'unkown file type: {file}')
    
    if delete:
        for file in add_files + [edf_file]:
            os.remove(file)
    return

if __name__=='__main__':
    documents = cfg.documents
    data = cfg.data
    
    
    
    files = ospath.list_files(data, exts=['edf'])
    edf_file = files[1]
