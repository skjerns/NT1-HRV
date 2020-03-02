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
from unisens import Unisens, SignalEntry, EventEntry, ValuesEntry
from unisens import CustomEntry
import os
from unisens.utils import read_csv
import shutil
import numpy as np
import ospath
import sleep_utils
import config as cfg
from tqdm import tqdm
import json_tricks



def to_unisens(edf_file, delete=False):
    name = ospath.basename(edf_file)[:-4]
    folder = ospath.dirname(edf_file)
    
    # get all additional files that belong to this EDF
    add_files = ospath.list_files(folder, patterns=name + '*')
    u = Unisens(ospath.join(folder, name), makenew=True, autosave=True)
    signal, shead, header = pyedflib.highlevel.read_edf(edf_file, ch_names='ECG I')
    annotations = header['annotations']
    
    sfreq = shead[0]['sample_rate']
    ecg_attrib={'data': signal.astype(np.float64),
                'sampleRate': sfreq,
                'ch_names': 'ECG',
                'lsbValue': '1',
                'unit': 'mV'}
    SignalEntry(id='ECG.bin', parent=u).set_data(**ecg_attrib)
    
    u.code = name
    u.sampling_frequency = sfreq
    u.duration = len(signal)//sfreq
    u.epochs = signal.shape[1]//int(u.sampling_frequency)//30
    
    annot_entry = EventEntry('annotations.csv', parent=u)
    annotations = [[int(a[0]*1000),a[2]]  for a in annotations]
    annot_entry.set_data(annotations, sampleRate=1000, typeLength=1, contentClass='Annotation')
    
    for file in add_files:
        if file.endswith('txt') or file.endswith('dat'):
            hypno = sleep_utils.read_hypnogram(file)
            times = np.arange(len(hypno))
            hypno = np.vstack([times, hypno]).T
            hypno_entry = EventEntry(id='hypnogram.csv', parent=u)
            hypno_entry.set_data(hypno, comment=f'File: {name}\nSleep stages 30s epochs.', 
                                 sampleRate=1/30, contentClass='Stage', typeLength=1)
            
        elif file.endswith('mat'):
            mat = mat73.loadmat(file)
            HRV = mat['Res']['HRV']
            feats_entry = CustomEntry('kubios.json', parent=u)
            feats_entry.set_data(HRV, comment='json dump of the kubios created RR file', fileType='JSON')
            
        elif file.endswith('npy'):
            art = np.load(file).ravel()
            times = np.arange(len(art))
            art = np.vstack([times, art]).T
            artefact_entry = ValuesEntry(id='artefacts.csv', parent=u)
            artefact_entry.set_data(art, sampleRate=1/15, dataType='int16')
            
        elif file.endswith('.edf'):
            pass
        else:
            raise Exception(f'unkown file type: {file}')
    
    u.save()
    
    if delete:
        for file in add_files + [edf_file]:
            os.remove(file)
    return

if __name__=='__main__':
    documents = cfg.documents
    data = cfg.data
    
    files = ospath.list_files(data, exts=['edf'])
    edf_file = 'Z:/NT1-HRV-data/228_06450.edf'
    to_unisens(edf_file)