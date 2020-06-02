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
import mat73
from unisens import SignalEntry, EventEntry, ValuesEntry
from sleep import CustomEntry
from pyedflib import highlevel
import numpy as np
from sleep import Patient
import ospath
import sleep_utils
import features
import misc
from datetime import datetime
from tqdm import tqdm
from joblib import Memory
import stimer
import subprocess

#%% We use memory to massively speed up these computations
memory = Memory(cfg.folder_cache, verbose=0)
read_edf_header = memory.cache(highlevel.read_edf_header)
read_edf = memory.cache(highlevel.read_edf)
loadmat = memory.cache(mat73.loadmat)

def to_unisens(edf_file, unisens_folder, overwrite=False, tqdm_desc= None,
               skip_exist=False):
    pass
#%% create unisens
    if tqdm_desc is None:  tqdm_desc=lambda x: None
    dtype = np.int16
    code = ospath.basename(edf_file)[:-4]
    folder = ospath.dirname(edf_file)

    unisens_folder = ospath.join(unisens_folder, code)
    
    if skip_exist and ospath.isdir(unisens_folder): return
        
    # get all additional files that belong to this EDF
    add_files = ospath.list_files(folder, patterns=code + '*')
    u = Patient(unisens_folder, makenew=False, autosave=True,
                measurementId=code)
    header = read_edf_header(edf_file)
    all_labels = header['channels']
    u.starttime = header['startdate']
    u.timestampStart = header['startdate'].strftime('%Y-%m-%dT%H:%M:%S')
    u.code = code
    
    attribs = misc.get_attribs()
    u.group = attribs[code].get('group', 'none')
    u.gender = attribs[code].get('gender', 'none')
    u.age = attribs[code].get('age', -1)
    u.match = attribs[code].get('match', '')
    u.channels = str(', '.join(header['channels']))
    u.startsec = (u.starttime.hour * 60 + u.starttime.minute) * 60 + u.starttime.second
        
    # if the ECG/EEG is broken, mark it
    edfs_ecg_broken = [p[1] for p in misc.read_csv(cfg.edfs_discard) if p[3]=='1']
    edfs_eeg_broken = [p[1] for p in misc.read_csv(cfg.edfs_discard) if p[4]=='1']
    
    # we need to see if the eeg/emg of this file can be used
    # if one of them is broken we also remove its match from analysis
    u.ecg_broken = (code in edfs_ecg_broken) or (u.match in edfs_ecg_broken)
    u.eeg_broken = (code in edfs_eeg_broken) or (u.match in edfs_eeg_broken)

    #%% #### add ECG ##########
    ########################
    tqdm_desc(f'{code}: Reading ECG')

    if not 'ECG' in u or overwrite:
        signals, shead, header = read_edf(edf_file, ch_names='ECG I', digital=True, verbose=False)
        signals[:,0:2]  = np.percentile(signals, 10), np.percentile(signals,90) # trick for viewer automatic scaling
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': 'ECG',
                    'lsbValue': lsb,
                    'baseline': offset,
                    'unit': 'mV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        
        SignalEntry(id='ECG.bin', parent=u).set_data(**attrib)
        
    
        u.sampling_frequency = shead[0]['sample_rate']
        u.duration = len(signals.squeeze())//shead[0]['sample_rate']
        u.epochs_signals = signals.shape[1]//int(u.sampling_frequency)//30  

    #%%#### add EEG ##########
    ##############################
    tqdm_desc(f'{code}: Reading EEG')
    if not 'EEG' in u or overwrite:
        chs = sleep_utils.infer_eeg_channels(all_labels)
        signals, shead, header = read_edf(edf_file, ch_names=chs, digital=True, verbose=False)
        if isinstance(signals, list): 
            signals = np.atleast_2d(signals[0])
            chs = chs[0]
        # trick for viewer automatic scaling
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) 
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': chs,
                    'lsbValue': lsb,
                    'baseline': offset,
                    'contentClass':'EEG',
                    'unit': 'uV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        SignalEntry(id='EEG.bin', parent=u).set_data(**attrib)

    #%%## add EOG #########
    #######################
    if not 'EOG' in u or overwrite:
        tqdm_desc(f'{code}: Reading EOG')
        chs = sleep_utils.infer_eog_channels(all_labels)
        signals, shead, header = read_edf(edf_file, ch_names=chs, digital=True, verbose=False)
        if isinstance(signals, list): 
            signals = np.atleast_2d(signals[0])
            chs = chs[0]
        # trick for viewer automatic scaling
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) 
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': chs,
                    'lsbValue': 1,
                    'baseline': 0,
                    'unit': 'uV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        SignalEntry(id='EOG.bin', parent=u).set_data(**attrib)
 
    #%%#### add EMG #########
    
    if not 'EMG' in u or overwrite:
        tqdm_desc(f'{code}: Reading EMG')
        chs = sleep_utils.infer_emg_channels(all_labels)
        if chs!=[]: # fix for 888_49272
            signals, shead, header = read_edf(edf_file, ch_names=chs, digital=True, verbose=False)
            if isinstance(signals, list): 
                signals = np.atleast_2d(signals[0])
                chs = chs[0]
            # trick for viewer automatic scaling
            signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) 
            pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
            dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
            
            lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
            attrib={'data': signals.astype(dtype), 
                        'sampleRate': shead[0]['sample_rate'],
                        'ch_names': chs,
                        'lsbValue': 1,
                        'baseline': 0,
                        'unit': 'uV',
                        'dmin': dmin,'dmax': dmax,
                        'pmin': pmin, 'pmax': pmax}
            SignalEntry(id='EMG.bin', parent=u).set_data(**attrib)
            
    #######################################
    #%%add Thorax #########
    ######################
    if not 'thorax' in u or overwrite:
        tqdm_desc(f'{code}: Reading Thorax')
        signals, shead, header = read_edf(edf_file, ch_names=['Thorax'], digital=True, verbose=False)
        # trick for viewer automatic scaling
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) 
        
        pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
        dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        attrib={'data': signals.astype(dtype), 
                    'sampleRate': shead[0]['sample_rate'],
                    'ch_names': 'thorax',
                    'lsbValue': 1,
                    'baseline': 0,
                    'unit': 'uV',
                    'dmin': dmin,'dmax': dmax,
                    'pmin': pmin, 'pmax': pmax}
        SignalEntry(id='thorax.bin', parent=u).set_data(**attrib)
        
    #######################################    
    #%% add Body / Lagesensor #########
    ########################################
    if (not 'body' in  u or overwrite) and 'Body' in all_labels:
        tqdm_desc(f'{code}: Reading Body')
        signals, shead, header = read_edf(edf_file, ch_names=['Body'], digital=True, verbose=False)
        signals[:,0:2] = np.percentile(signals, 10), np.percentile(signals,90) 
            
        if np.ptp(signals)<10: # we have some weird body positions that we cant decode
        
            pmin, pmax = shead[0]['physical_min'], shead[0]['physical_max']
            dmin, dmax = shead[0]['digital_min'], shead[0]['digital_max']
            
            comment = 'Lagesensor: 1 = Bauchlage, 2 = aufrecht, 3 = links, 4 = rechts,' \
                      '5 = aufrecht (Kopfstand), 6 = Rückenlage'
            
            lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
            attrib={'data': signals.astype(dtype), 
                        'sampleRate': shead[0]['sample_rate'],
                        'ch_names': 'body',
                        'lsbValue': 1,
                        'baseline': 0,
                        'unit': 'uV',
                        'dmin': dmin,'dmax': dmax,
                        'pmin': pmin, 'pmax': pmax,
                        'comment': comment}
            SignalEntry(id='body.bin', parent=u).set_data(**attrib)
    
    #%% add annotations #######
    ################################
    if not 'annotations' in u or overwrite:
        annotations = header['annotations']
        if annotations!=[]:
            annot_entry = EventEntry('annotations.csv', parent=u)
            annotations = [[int(a[0]*1000),a[2]]  for a in annotations]
            annot_entry.set_data(annotations, sampleRate=1000, typeLength=1, contentClass='Annotation')
 
    #%%#### add rest #######
    ############################
    for file in add_files:
        #%% add arousals
        if file.endswith('_arousals.txt'):
            if  'arousals' in u and not overwrite: continue
            lines = misc.read_csv(file, convert_nums=True)
            
            sdate = u.starttime
            data = []
            for t_arousal, length, _ in lines[4:]:
                t_arousal = f'{sdate.year}.{sdate.month}.{sdate.day} ' + t_arousal[:8]
                t_arousal = datetime.strptime(t_arousal, '%Y.%m.%d %H:%M:%S')
                epoch = (t_arousal - sdate).seconds//30
                data += [[epoch, length]]
            
            arousal_event = EventEntry('arousals.csv', parent=u)
            arousal_event.set_data(data, comment=f'Arousal appearance epoch, name is lengths in seconds', 
                                 sampleRate=1/30, contentClass='Arousal', typeLength=1)
        #%% add hypnogram
        elif file.endswith('txt'):
            if  'hypnogram' in u and not overwrite: continue
            tqdm_desc(f'{code}: Reading Hypnogram')
            hypno = sleep_utils.read_hypnogram(file)
            u.epochs_hypno = len(hypno)
            times = np.arange(len(hypno))
            hypno = np.vstack([times, hypno]).T
            hypno_entry = EventEntry(id='hypnogram.csv', parent=u)
            hypno_entry.set_data(hypno, comment=f'File: {code}\nSleep stages 30s epochs.', 
                                 sampleRate=1/30, contentClass='Stage', typeLength=1)
        
        elif file.endswith('.hypno'):
            if  'hypnogram_old' in u and not overwrite: continue
            hypno = sleep_utils.read_hypnogram(file)
            if not hasattr(u, 'epochs_hypno'): u.epochs_hypno = len(hypno)
            times = np.arange(len(hypno))
            hypno = np.vstack([times, hypno]).T
            hypno_old_entry = EventEntry(id='hypnogram_old.csv', parent=u)
            hypno_old_entry.set_data(hypno, comment=f'File: {code}\nSleep stages 30s epochs.', 
                                 sampleRate=1/30, contentClass='Stage', typeLength=1)
        #%% add kubios
        elif file.endswith('mat'):     
            if  'feats.pkl' in u and not overwrite: continue
            tqdm_desc(f'{code}: Reading Kubios')
            mat = loadmat(file)
            HRV = mat['Res']['HRV']

            feats_entry = CustomEntry('feats.pkl', parent=u)
            feats_entry.set_data(HRV, comment='pickle dump of the kubios created features file', fileType='pickle')
        
            for nr, name in cfg.mapping_feats.items():
                # if there is no function for this feature name
                # we skip the calculation of this feature
                # it might not be implemented yet.
                if not name in features.__dict__: continue
                wsize = cfg.default_wsize
                step = cfg.default_step
                offset = cfg.default_offset
                u.get_feat(name, wsize=wsize, step=step, offset=offset)
            u.get_artefacts(wsize=wsize, step=step, offset=offset)

        
        
        #%% add artefact
        ############ removed artefact detection and calculated from kubios above
        # elif file.endswith('npy'):
        #     if  'artefacts' in u and not overwrite: continue
        #     tqdm_desc(f'{code}: Reading artefacts')
        #     art = np.load(file).ravel()
        #     u.epochs_art = len(art)//2
        #     u.artefact_percentage = np.mean(art)
        #     times = np.arange(len(art))
        #     art = np.vstack([times, art]).T
        #     artefact_entry = ValuesEntry(id='artefacts.csv', parent=u)
        #     artefact_entry.set_data(art, sampleRate=1/15, dataType='int16')
            
        elif file.endswith('.edf'):
            pass

        else:
            raise Exception(f'unkown file type: {file}')   

    u.save()

#%% main

if __name__=='__main__':
    from joblib import Parallel, delayed
    documents = cfg.documents
    data = cfg.folder_edf
    unisens_folder = cfg.folder_unisens
    
    mat_files = ospath.list_files(data, exts=['mat'])
    
    files = ospath.list_files(data, exts=['edf'])
    
    execute = True
    if len(mat_files)<len(files):
        answer = input(f'{len(files)-len(mat_files)} edf files have no MAT file. '
                  f'Please copy .mat-files to folder. \nContinue anyway? (Y/N)\n')
        if not 'Y' in answer.upper():
            execute = False
    if execute:
        Parallel(n_jobs=5, batch_size=4)(delayed(to_unisens)(
            edf_file, unisens_folder=unisens_folder, skip_exist=False, overwrite=False) for edf_file in tqdm(files, desc='Converting'))