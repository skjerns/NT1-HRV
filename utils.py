# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:19:26 2019

@author: skjerns
"""
import warnings
import ospath #pip install https://github.com/skjerns/skjerns-utils
import numpy as np
import pyedflib #pip install https://github.com/skjerns/pyedflib/archive/custom_version.zip
from tqdm import tqdm


def read_pyedf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=True):
    """
    Reading EDF+ data with pyedflib.

    Will load the edf and return the signals, the headers of the signals 
    and the header of the EDF
        
    :param edf_file: link to an edf file
    :param ch_nrs: The numbers of channels to read (optional)
    :param ch_names: The names of channels to read (optional)
    :returns: signals, signal_headers, header
    """      
    assert (ch_nrs is  None) or (ch_names is None), 'names xor numbers should be supplied'
    if ch_nrs is not None and not isinstance(ch_nrs, list): ch_nrs = [ch_nrs]
    if ch_names is not None and not isinstance(ch_names, list): ch_names = [ch_names]

    with pyedflib.EdfReader(edf_file) as f:
        # see which channels we want to load
        available_chs = [ch.upper() for ch in f.getSignalLabels()]
        
        # find out which number corresponds to which channel
        if ch_names is not None:
            ch_nrs = []
            for ch in ch_names:
                if not ch.upper() in available_chs:
                    warnings.warn('{} is not in source file (contains {})'.format(ch, available_chs))
                    print('will be ignored.')
                else:    
                    ch_nrs.append(available_chs.index(ch.upper()))
                    
        # if there ch_nrs is not given, load all channels            
        if ch_nrs is None: # no numbers means we load all
            n = f.signals_in_file
            ch_nrs = np.arange(n)
            
        # load headers, signal information and 
        header = f.getHeader()
        signal_headers = [f.getSignalHeaders()[c] for c in ch_nrs]

        signals = []
        for i,c in enumerate(tqdm(ch_nrs, desc='Reading Channels', disable=not verbose)):
            signal = f.readSignal(c, digital=digital)
            signals.append(signal)
 
        # we can only return a np.array if all signals have the same samplefreq           
        sfreqs = [header['sample_rate'] for header in signal_headers]
        all_sfreq_same = sfreqs[1:]==sfreqs[:-1]
        if all_sfreq_same:
            dtype = np.int if digital else np.float
            signals = np.array(signals, dtype=dtype)
        elif verbose:
            warnings.warn('Not all sampling frequencies are the same ({}). '.format(sfreqs))    
    assert len(signals)==len(signal_headers), 'Something went wrong, lengths of headers is not length of signals'
    return  signals, signal_headers, header







