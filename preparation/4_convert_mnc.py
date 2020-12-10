# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:35:46 2020

This file converts all files that are we have info from the mnc to Unisens

@author: Simon
"""

import tempfile
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

# We use memory to massively speed up these computations
memory = Memory(cfg.folder_cache, verbose=0)
read_edf_header = memory.cache(highlevel.read_edf_header)
read_edf = memory.cache(highlevel.read_edf)
loadmat = memory.cache(mat73.loadmat)


def repair_file(edf_file):
    """
    some of the SSC corohort files were broken, so we need to fix them
    manually. It is a rather easy fix in which an NA was written instead of a
    timestamp. We replace it with a generic date that points to 10:10:10.
    """
    print(f'attempting to repair {edf_file}')
    with open(edf_file, 'rb') as f:
        line = next(f) # grab first line
        old = b'NA      '
        new = b'10.10.10' # padded with spaces to make same length as old
        line = line.replace(old, new)

    with open(edf_file, 'rb+') as f:
        f.seek(0) # move file pointer to beginning of file
        f.write(line)


def to_unisens(edf_file, unisens_folder, mat_folder, overwrite=False, skip_exist=False):

    dtype = np.int16
    folder = ospath.dirname(edf_file)
    filename = ospath.basename(edf_file)[:-9] # remove "-nsrr.edf" from filename

    mnc_info = misc.get_mnc_info()
    try:
        attribs = mnc_info[filename.upper().replace(' ', '_')]
    except:
        print(f'Info for {filename.upper().replace(" ", "_")} not found')
        return

    # get all additional files that belong to this EDF
    patterns = [filename + '*.xml', filename + '*.sta']
    add_files = ospath.list_files(folder, patterns=patterns)
    if len(add_files)==0:
        print(f'No hypnogram for {filename}, skip')
        return

    # try to find mat files
    mat_files = ospath.list_files(mat_folder, patterns=[filename + '*.mat'])
    if len(mat_files)==1:
        mat_file=mat_files[0]
    else:
        print(f'No matfile found for {filename}')
        misc.extract_ecg(edf_file, 'z:/ecg/')
        return

    # get the codified version of this file
    code = misc.codify(filename)
    unisens_folder = ospath.join(unisens_folder, code)

    # if this unisens folder exists, skip if requested
    if skip_exist and ospath.isdir(unisens_folder): return
        

    # now create the meta information for the new file
    try:
        header = read_edf_header(edf_file)
    except:
        repair_file(edf_file)
        try:
            header = read_edf_header(edf_file)
        except  Exception as e:
            print(f'cant load {filename}, broken edf {e}')
            return
    channels = header['channels']

    chs = [ch for ch in channels if 'ECG' in ch.upper()]
    if 'cs_ECG' in chs and len(chs)>1:
        chs.remove('cs_ECG')


    # add metadata for this file
    u = Patient(unisens_folder, makenew=True, autosave=True, measurementId=code)
    u.starttime = header['startdate']
    u.timestampStart = header['startdate'].strftime('%Y-%m-%dT%H:%M:%S')
    u.code = code
    u.dataset = 'mnc'
    u.channels = str(', '.join(channels))
    u.startsec = (u.starttime.hour * 60 + u.starttime.minute) * 60 + u.starttime.second
    if u.startsec==0:print(edf_file)
    u.DQ0602 = attribs['DQ0602']
    u.hypocretin = attribs['CSF hypocretin-1']
    u.label = attribs['Label']
    u.cohort = attribs['Cohort']

    diagnosis = attribs['Diagnosis']
    if 'CONTROL' in diagnosis:
        group = 'control'
    elif 'T1' in diagnosis:
        group = 'nt1'
    elif 'OTHER HYPERSOMNIA' in diagnosis:
        group = 'hypersomnia'
    else:
        group = attribs['Diagnosis']
        raise AttributeError(f'unkown group: {group} for {filename}')
    u.group = group

    # %% Add ECG channel

    # add the original ECG channel
    sig_orig, shead_orig, _ = read_edf(edf_file, ch_names=chs[0], verbose=False, digital=True)
    assert sig_orig.max()<=32767 and sig_orig.min()>=-32768, 'min/max exceeds int16'
    pmin, pmax = shead_orig[0]['physical_min'], shead_orig[0]['physical_max']
    dmin, dmax = shead_orig[0]['digital_min'], shead_orig[0]['digital_max']
    lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
    attrib={'data': sig_orig.astype(dtype),
            'sampleRate': shead_orig[0]['sample_rate'],
            'ch_names': 'ECG',
            'lsbValue': lsb,
            'baseline': offset,
            'unit': 'mV',
            'dmin': dmin,'dmax': dmax,
            'pmin': pmin, 'pmax': pmax}
    u.sampling_frequency = shead_orig[0]['sample_rate']
    SignalEntry(id='ECG.bin', parent=u).set_data(**attrib)

    # %% now extract the RR intervals

    if not 'annotations' in u or overwrite:
        annotations = header['annotations']
        if annotations!=[]:
            annot_entry = EventEntry('annotations.csv', parent=u)
            annotations = [[int(a[0]*1000),a[2]]  for a in annotations]
            annot_entry.set_data(annotations, sampleRate=1000, typeLength=1, contentClass='Annotation')

    # %% add hypnogram, if it is available
    assert len(add_files)!=1, 'Only one hypnogram file? seems weird'
    if len(add_files)==2:
        hypnograms = [sleep_utils.read_hypnogram(file, epochlen_infile=30 if file.endswith('annot') else None) for file in add_files]
        try:
            np.testing.assert_array_almost_equal(hypnograms[0], hypnograms[1], err_msg=f'{add_files}')
        except:
            print(f'files not equal: {add_files}')

        if not 'hypnogram' in u or  overwrite:
            hypno = hypnograms[0]
            u.epochs_hypno = len(hypno)
            times = np.arange(len(hypno))
            hypno = np.vstack([times, hypno]).T
            hypno_entry = EventEntry(id='hypnogram.csv', parent=u)
            hypno_entry.set_data(hypno, comment=f'File: {code}\nSleep stages 30s epochs.', 
                                 sampleRate=1/30, contentClass='Stage', typeLength=1)


    # %% Add features
    if not 'feats.pkl' in u or overwrite:
        mat = loadmat(mat_file)
        HRV = mat['Res']['HRV']
    
        feats_entry = CustomEntry('feats.pkl', parent=u)
        feats_entry.set_data(HRV, comment='pickle dump of the kubios created features file', fileType='pickle')
    
        wsize = cfg.default_wsize
        step = cfg.default_step
        offset = True
        u.compute_features(offset=False)
        u.get_artefacts(wsize=wsize, step=step, offset=False)

        rri_entry = CustomEntry('RRi.pkl', parent=u)
        rri_entry.set_data(HRV['Data']['RRi'], comment='raw data of RRi, the interpolated RRs at 4hz', fileType='pickle')
        rri_entry.sampleRate = 4
    u.save()
    return True

# %% main

if __name__=='__main__':
    from joblib import Parallel, delayed
    documents = cfg.documents
    data = cfg.folder_mnc
    unisens_folder = cfg.folder_unisens
    mat_folder = cfg.folder_mat

    # only get files called -nsrr.edf, ignore the others
    files = ospath.list_files(data, exts=['nsrr.edf'], subfolders=True)

    edf_file = files[0]

    Parallel(n_jobs=1)(delayed(to_unisens)(
        edf_file, unisens_folder=unisens_folder, mat_folder=mat_folder,
        skip_exist=True, overwrite=False) for edf_file in tqdm(files, desc='Converting'))


    # %% print info
    input('press enter to check which file/info is missing')
    info = misc.get_mnc_info()
    fullfiles = files.copy()
    files = [f[:-9] for f in files] # remove '-nsrr.edf'
    files = [ospath.basename(file.upper()).replace(' ', '_') for file in files]

    nt1 = []
    hyp = []
    cnt = []
    included = {}
    missing_file = []
    missing_info = []
    to_extract = []
    for name, full in zip(files, fullfiles):
        if name in info:
            included[name] = info[name].copy()
            if 'CONTROL' in included[name]['Diagnosis']:
                cnt.append(included[name])
                to_extract.append(full)
            elif 'T1' in included[name]['Diagnosis']:
                nt1.append(included[name])
                to_extract.append(full)

            elif 'OTHER HYPERSOMNIA' in included[name]['Diagnosis']:
                hyp.append(included[name])
            del info[name]
        else:
            missing_info.append(full)

    for name, value in info.items():
        missing_file.append(f'{info[name]["Cohort"].lower()}/{info[name]["ID"]}')



    # Parallel(n_jobs=10)(delayed(to_unisens)(
            # edf_file, unisens_folder=unisens_folder, skip_exist=False, overwrite=False) for edf_file in tqdm(files, desc='Converting'))

    # single process
    # for file in tqdm(files):
        # to_unisens(file, unisens_folder=unisens_folder, skip_exist=False, overwrite=False)
