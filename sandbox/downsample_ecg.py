# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:03:27 2020

downsample the eeg of an edf


@author: skjerns
"""
import os, sys
import numpy as np
from tqdm import tqdm
import mne
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:56:31 2018
@author: skjerns
Gist to save a mne.io.Raw object to an EDF file using pyEDFlib
(https://github.com/holgern/pyedflib)
Disclaimer:
    - Saving your data this way will result in slight 
      loss of precision (magnitude +-1e-09).
    - It is assumed that the data is presented in Volt (V), 
      it will be internally converted to microvolt
    - BDF or EDF+ is selected based on the filename extension
    - Annotations are lost in the process.
      Let me know if you need them, should be easy to add.
"""

import pyedflib # pip install pyedflib
from pyedflib import highlevel # new high-level interface
from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta
import mne

#%% create downsampled files
def resample(raw, o_sfreq, t_sfreq):
    """
    resample a signal using MNE resample functions
    This automatically is optimized for EEG applying filters etc
    
    :param raw:     a 1D data array
    :param o_sfreq: the original sampling frequency
    :param t_sfreq: the target sampling frequency
    :returns: the resampled signal
    """
    if o_sfreq==t_sfreq: return raw
    raw = np.atleast_2d(raw)
    ch_names=['ch{}'.format(i) for i in range(len(raw))]
    info = mne.create_info(ch_names=ch_names, sfreq=o_sfreq, ch_types=['eeg'])
    raw_mne = mne.io.RawArray(raw, info, verbose='ERROR')
    resampled = raw_mne.resample(t_sfreq, n_jobs=3)
    new_raw = resampled.get_data().squeeze()
    return new_raw.astype(raw.dtype, copy=False)


sigs = {}

for sfreq in tqdm([128, 200,256, 250, 350,512]):
    signal, sheaders, header = highlevel.read_edf('Z:/ecg/110_88246.edf')

    sigs[sfreq] = np.atleast_2d (resample(signal, 512, sfreq))
    
    # rescale signal to match max
    for head in sheaders:
        head['sample_rate'] = sfreq
        head['physical_min'] = sigs[sfreq].min()
        head['physical_max'] = sigs[sfreq].max()
        
    highlevel.write_edf(f'z:/ecg/512to{sfreq}hz.edf', sigs[sfreq], sheaders, header)
    
    
# create upsampled file 
sigs = {}

for sfreq in tqdm([128, 200,256,512]):
    signal, sheaders, header = highlevel.read_edf('Z:/ecg/375_90431.edf')

    sigs[sfreq] = np.atleast_2d (resample(signal, 512, sfreq))
    
    # rescale signal to match max
    for head in sheaders:
        head['sample_rate'] = sfreq
        head['physical_min'] = sigs[sfreq].min()
        head['physical_max'] = sigs[sfreq].max()
        
    highlevel.write_edf(f'z:/ecg/200to{sfreq}hz.edf', sigs[sfreq], sheaders, header)
    
    
#%% check features