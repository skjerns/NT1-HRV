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
from sleep import FeaturesEntry
import shutil
import numpy as np
from sleep import Patient
import ospath
import sleep_utils
import config as cfg
from tqdm import tqdm
import json_tricks
import misc




def to_unisens(edf_file, unisens_folder=None, overwrite=False, tqdm_desc= None):
    pass
#%%    
    if tqdm_desc is None:  tqdm_desc=lambda x: None
    dtype = np.int16
    code = ospath.basename(edf_file)[:-4]
    folder = ospath.dirname(edf_file)
    if unisens_folder is None: 
        unisens_folder = '.'
    
    unisens_folder = ospath.join(unisens_folder, code)
        
    # get all additional files that belong to this EDF
    add_files = ospath.list_files(folder, patterns=code + '*')
    u = Patient(unisens_folder, makenew=False, autosave=True)
    header = highlevel.read_edf_header(edf_file)
    all_labels = header['channels']
    u.starttime = header['startdate']
    u.code = code
    
    attribs = misc.get_attribs()
    u.group = attribs[code].get('group', 'none')
    u.gender = attribs[code].get('gender', 'none')
    u.age = attribs[code].get('age', -1)
    

    #%%####################
    #### add ECG ##########
    tqdm_desc(f'{code}: Reading ECG')
    if not 'ECG' in u or overwrite:
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
        u.duration = len(signals.squeeze())//shead[0]['sample_rate']
        u.epochs_signals = signals.shape[1]//int(u.sampling_frequency)//30        
    #%%####################
    #### add EEG ##########
    tqdm_desc(f'{code}: Reading EEG')
    if not 'EEG' in u or overwrite:
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
    if not 'EOG' in u or overwrite:
        tqdm_desc(f'{code}: Reading EOG')
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
    if not 'EMG' in u or overwrite:
        tqdm_desc(f'{code}: Reading EMG')
        emg = sleep_utils.infer_emg_channels(all_labels)
        if emg!=[]: # fix for 888_49272
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
            emg_entry = SignalEntry(id='EMG.bin', parent=unisens_folder).set_data(**eog_attrib)

    #%%####################
    #### add annotations #######
    if not 'annotations' in u or overwrite:
        annotations = header['annotations']
        if annotations!=[]:
            annot_entry = EventEntry('annotations.csv', parent=unisens_folder)
            annotations = [[int(a[0]*1000),a[2]]  for a in annotations]
            annot_entry.set_data(annotations, sampleRate=1000, typeLength=1, contentClass='Annotation')
            
    #%%####################
    #### add rest #######
    
    for file in add_files:
        if file.endswith('txt') or file.endswith('dat'):
            if  'hypnogram' in u and not overwrite: continue
            tqdm_desc(f'{code}: Reading Hypnogram')
            hypno = sleep_utils.read_hypnogram(file)
            u.epochs_hypno = len(hypno)
            times = np.arange(len(hypno))
            hypno = np.vstack([times, hypno]).T
            hypno_entry = EventEntry(id='hypnogram.csv', parent=unisens_folder)
            hypno_entry.set_data(hypno, comment=f'File: {code}\nSleep stages 30s epochs.', 
                                 sampleRate=1/30, contentClass='Stage', typeLength=1)

        elif file.endswith('mat'):              
            tqdm_desc(f'{code}: Reading Kubios')
            mat = mat73.loadmat(file)
            HRV = mat['Res']['HRV']
            startsecond = (u.starttime.hour * 60 + u.starttime.minute) * 60 + u.starttime.second
            T_RR = HRV['Data']['T_RR'].squeeze() - startsecond
            T_RR = list(zip(T_RR, ['RR']*len(T_RR)))
            if not 'T_RR' in u:
                rr_entry = ValuesEntry(id='T_RR.csv', parent=unisens_folder)
                rr_entry.set_data(T_RR, ch_names='RR')
                           
            if 'feats' in u:
                feats_entry = u.feats
            else:
                feats_entry = FeaturesEntry('feats.json', parent=unisens_folder)
                feats_entry.set_data(HRV, comment='json dump of the kubios created RR file', fileType='JSON')
            for key in HRV['TimeVar']:
                if key=='Overview':continue
                CustomEntry('feats/' + key + '.npy', parent=feats_entry)\
                    .set_data(HRV['TimeVar'][key])
        
        elif file.endswith('npy'):
            if  'artefacts' in u and not overwrite: continue
            tqdm_desc(f'{code}: Reading artefacts')
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
    if 'rr_entry' in locals(): u.add_entry(rr_entry)
    if 'artefact_entry' in locals(): u.add_entry(artefact_entry)
    if 'hypno_entry' in locals(): u.add_entry(hypno_entry)
    if 'feats_entry' in locals():u.add_entry(feats_entry)
    if 'annot_entry' in locals(): u.add_entry(annot_entry)
    
    u.save()

#%%
if __name__=='__main__':
    
    documents = cfg.documents
    data = cfg.folder_edf
    unisens = cfg.folder_unisens
    
    files = ospath.list_files(data, exts=['edf'])
    
    progbar = tqdm(files)
    for edf_file in progbar:
        to_unisens(edf_file,unisens_folder=unisens, tqdm_desc= progbar.set_description)
        