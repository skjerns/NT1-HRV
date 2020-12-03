# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:35:46 2020

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

memory = Memory(cfg.folder_cache, verbose=0)
read_edf_header = memory.cache(highlevel.read_edf_header)
read_edf = memory.cache(highlevel.read_edf)
loadmat = memory.cache(mat73.loadmat)


#%% We use memory to massively speed up these computations

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


def to_unisens(edf_file, unisens_folder, overwrite=False, skip_exist=False):

    dtype = np.int16
    filename = ospath.basename(edf_file)[:-9] # remove "-nsrr.edf" from filename
    mnc_info = misc.get_mnc_info()
    try:
        attribs = mnc_info[filename.upper()]
    except:
        # print(f'Info for {edf_file} not found')
        return

    # get the codified version of this file
    code = misc.codify(filename)
    folder = ospath.dirname(edf_file)
    unisens_folder = ospath.join(unisens_folder, code)

    # if this unisens folder exists, skip if requested
    if skip_exist and ospath.isdir(unisens_folder): return
        
    # get all additional files that belong to this EDF
    add_files = ospath.list_files(folder, patterns=[filename + '*.xml', filename + '*.eannot'])
    u = Patient(unisens_folder, makenew=True, autosave=True, measurementId=code)

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


    # add metadata for this file
    channels = header['channels']
    u.starttime = header['startdate']
    u.timestampStart = header['startdate'].strftime('%Y-%m-%dT%H:%M:%S')
    u.code = code
    u.dataset = 'mnc'
    u.channels = str(', '.join(channels))
    u.startsec = (u.starttime.hour * 60 + u.starttime.minute) * 60 + u.starttime.second
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

    chs = [ch for ch in channels if 'ECG' in ch.upper()]
    if 'cs_ECG' not in chs:
        print('resampled channel not found, skipping for now. Maybe resample in later version?')
        return
    else:
        chs.remove('cs_ECG')


    # first add the resampled channel that should be at 100 Hz
    sig_cs, shead_cs, _ = read_edf(edf_file, ch_names='cs_ECG', verbose=False, digital=True)
    assert sig_cs.max()<=32767 and sig_cs.min()>=-32768, 'min/max exceeds int16'
    pmin, pmax = shead_cs[0]['physical_min'], shead_cs[0]['physical_max']
    dmin, dmax = shead_cs[0]['digital_min'], shead_cs[0]['digital_max']
    lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
    attrib={'data': sig_cs.astype(dtype),
            'sampleRate': shead_cs[0]['sample_rate'],
            'ch_names': 'ECG',
            'lsbValue': lsb,
            'baseline': offset,
            'unit': 'mV',
            'dmin': dmin,'dmax': dmax,
            'pmin': pmin, 'pmax': pmax}
    u.sfreq = shead_cs[0]['sample_rate']
    SignalEntry(id='ECG.bin', parent=u).set_data(**attrib)

    # now add the original
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
    u.sfreq_orig = shead_orig[0]['sample_rate']
    SignalEntry(id='ECG_orig.bin', parent=u).set_data(**attrib)

    #%% now extract the RR intervals



    #%% add hypnogram, if it is available
    assert len(add_files)!=1, 'Only one hypnogram file? seems weird'
    if len(add_files)==2:
        hypnograms = [sleep_utils.read_hypnogram(file, epochlen_infile=30) for file in add_files]
        try:
            np.testing.assert_array_almost_equal(hypnograms[0], hypnograms[1], err_msg=f'{add_files}')
        except:
            print(f'files not equal: {add_files}')

    u.save()

#%% main

if __name__=='__main__':
    from joblib import Parallel, delayed
    documents = cfg.documents
    data = cfg.folder_mnc
    unisens_folder = cfg.folder_unisens

    files = ospath.list_files(data, exts=['edf'], subfolders=True)

    edf_file = files[0]

    # for edf_file in tqdm(files):
    #     to_unisens(edf_file, unisens_folder)

    input('press enter to check which file/info is missing')
    info = misc.get_mnc_info()
    fullfiles = files.copy()
    files = [f[:-9] for f in files]
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
