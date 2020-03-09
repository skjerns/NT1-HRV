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
import config as cfg
import shutil
import mat73
import pyedflib
import unisens
from unisens import Unisens, SignalEntry, EventEntry, ValuesEntry
from unisens import CustomEntry
import os
from pyedflib import highlevel
from unisens.utils import read_csv
import shutil
import numpy as np
import ospath
import sleep_utils
import config as cfg
from tqdm import tqdm
import json_tricks




def to_unisens(edf_file, delete=False, overwrite=False):
    pass
#%%    
    dtype = np.int16
    name = ospath.basename(edf_file)[:-4]
    folder = ospath.dirname(edf_file)
    unisens_folder = ospath.join(folder, name)
        
    # get all additional files that belong to this EDF
    add_files = ospath.list_files(folder, patterns=name + '*')
    u = Unisens(unisens_folder, makenew=True, autosave=True)
    header = highlevel.read_edf_header(edf_file)
    all_labels = header['channels']
    u.startdate = header['startdate']
    u.code = name    
    print('post')

    #%%####################
    #### add ECG ##########
    if not ospath.exists(ospath.join(folder, 'ECG.bin')) or overwrite:
        signals, shead, header = pyedflib.highlevel.read_edf(edf_file, ch_names='ECG I', digital=True)
        signals[:,0:2]  = np.percentile(signals, 10), np.percentile(signals,90) # trick for viewer automatic scaling
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        ecg_attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': 'ECG',
                    'lsbValue': lsb,
                    'baseline': offset,
                    'unit': 'mV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        
        ecg_entry = SignalEntry(id='ECG.bin', parent=unisens_folder).set_data(**ecg_attrib)
        

        u.sampling_frequency = shead[0]['sample_rate']
        u.duration = len(signals)//shead[0]['sample_rate']
        u.epochs_signals = signals.shape[1]//int(u.sampling_frequency)//30        
    #%%####################
    #### add EEG ##########
    if not ospath.exists(ospath.join(folder, 'EEG.bin')) or overwrite:
        eeg = sleep_utils.infer_eeg_channels(all_labels)
        signals, shead, header = highlevel.read_edf(edf_file, ch_names=eeg, digital=True)
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) # trick for viewer automatic scaling
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        eeg_attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': eeg,
                    'lsbValue': lsb,
                    'baseline': offset,
                    'contentClass':'EEG',
                    'unit': 'uV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        eeg_entry = SignalEntry(id='EEG.bin', parent=unisens_folder).set_data(**eeg_attrib)

 
    #%%####################
    #### add EOG #########
    if not ospath.exists(ospath.join(folder, 'EOG.bin')) or overwrite:
        eog = sleep_utils.infer_eog_channels(all_labels)
        signals, shead, header = highlevel.read_edf(edf_file, ch_names=eog, digital=True)
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) # trick for viewer automatic scaling
        
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        eog_attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': eog,
                    'lsbValue': 1,
                    'baseline': 0,
                    'unit': 'uV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        eog_entry = SignalEntry(id='EOG.bin', parent=unisens_folder).set_data(**eog_attrib)
        
        
    #%%####################
    #### add EMG #########
    if not ospath.exists(ospath.join(folder, 'EMG.bin')) or overwrite:
        emg = sleep_utils.infer_eog_channels(all_labels)
        signals, shead, header = highlevel.read_edf(edf_file, ch_names=emg, digital=True)
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) # trick for viewer automatic scaling
        
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        eog_attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': emg,
                    'lsbValue': 1,
                    'baseline': 0,
                    'unit': 'uV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        eog_entry = SignalEntry(id='EMG.bin', parent=unisens_folder).set_data(**eog_attrib)

    #%%####################
    #### add annotations #######
    if not ospath.exists(ospath.join(folder, 'annotations.csv')) or overwrite:
        annotations = header['annotations']
        if annotations!=[]:
            annot_entry = EventEntry('annotations.csv', parent=unisens_folder)
            annotations = [[int(a[0]*1000),a[2]]  for a in annotations]
            annot_entry.set_data(annotations, sampleRate=1000, typeLength=1, contentClass='Annotation')
            
    #%%####################
    #### add rest #######
    
    for file in add_files:
        if file.endswith('txt') or file.endswith('dat'):
            if ospath.exists(ospath.join(folder, 'hypnogram.csv')) and not overwrite:continue
            hypno = sleep_utils.read_hypnogram(file)
            u.epochs_hypno = len(hypno)
            times = np.arange(len(hypno))
            hypno = np.vstack([times, hypno]).T
            hypno_entry = EventEntry(id='hypnogram.csv', parent=unisens_folder)
            hypno_entry.set_data(hypno, comment=f'File: {name}\nSleep stages 30s epochs.', 
                                 sampleRate=1/30, contentClass='Stage', typeLength=1)

        elif file.endswith('mat'):
            if ospath.exists(ospath.join(folder, 'kubios.json')) and not overwrite:continue
            mat = mat73.loadmat(file)
            HRV = mat['Res']['HRV']
            feats_entry = CustomEntry('kubios.json', parent=unisens_folder)
            feats_entry.set_data(HRV, comment='json dump of the kubios created RR file', fileType='JSON')
        elif file.endswith('npy'):
            if ospath.exists(ospath.join(folder, 'artefacts.csv')) and not overwrite:continue
            art = np.load(file).ravel()
            u.epochs_art = len(art)//2
            times = np.arange(len(art))
            art = np.vstack([times, art]).T
            artefact_entry = ValuesEntry(id='artefacts.csv', parent=unisens_folder)
            artefact_entry.set_data(art, sampleRate=1/15, dataType='int16')
            
        elif file.endswith('.edf'):
            pass
        else:
            raise Exception(f'unkown file type: {file}')
    #%%####################
    # we add the entries manually so they are in the right order in the unisens viewer.
    if 'ecg_entry' in locals(): u.add_entry(ecg_entry)
    if 'eeg_entry' in locals(): u.add_entry(eeg_entry)
    if 'eog_entry' in locals(): u.add_entry(eog_entry)
    if 'emg_entry' in locals(): u.add_entry(emg_entry)
    if 'artefact_entry' in locals(): u.add_entry(artefact_entry)
    if 'hypno_entry' in locals(): u.add_entry(hypno_entry)
    if 'feats_entry' in locals(): u.add_entry(feats_entry)
    if 'annot_entry' in locals(): u.add_entry(annot_entry)
    u.save()
    
    if delete:
        for file in add_files + [edf_file]:
            os.remove(file)
    return
#%%
if __name__=='__main__':
    
    documents = cfg.documents
    data = cfg.data
    
    files = ospath.list_files(data, exts=['edf'])
    for edf_file in tqdm(files):
        to_unisens(edf_file)