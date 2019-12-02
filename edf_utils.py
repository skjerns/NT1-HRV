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



def read_edf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=True):
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
        n_chrs = f.signals_in_file

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
            ch_nrs = np.arange(n_chrs)
        
        # convert negative numbers into positives
        ch_nrs = [n_chrs+ch if ch<0 else ch for ch in ch_nrs]
        
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



def write_edf(edf_file, signals, signal_headers, header, digital=False):
    """
    Write signals to an edf_file. Header can be generated on the fly.
    
    :param signals: The signals as a list of arrays or a ndarray
    :param signal_headers: a list with one signal header (dict) for each signal.
                           See pyedflib.EdfWriter.setSignalHeader
    :param header: a main header (dict) for the EDF file, see 
                   pyedflib.EdfWriter.setHeader for details

    :param digital: whether signals are presented digitally or in physical values
    """
    assert header is None or isinstance(header, dict), 'header must be dictioniary'
    assert isinstance(signal_headers, list), 'signal headers must be list'
    assert len(signal_headers)==len(signals), 'signals and signal_headers must be same length'
        
    n_channels = len(signals)
    
    with pyedflib.EdfWriter(edf_file, n_channels=n_channels) as f:  
        f.setSignalHeaders(signal_headers)
        f.setHeader(header)
        f.writeSamples(signals, digital=digital)
        
    return ospath.isfile(edf_file) 

def read_edf_header(edf_file):
    """
    Reads the header and signal headers of an EDF file
    
    :returns: header of the edf file (dict)
    """
    with pyedflib.EdfReader(edf_file) as f:
        summary = f.getHeader()
        summary['Duration'] = f.getFileDuration
        summary['SignalHeaders'] = f.getSignalHeaders()
        summary['channels'] = f.getSignalLabels()
    del f
    return summary


def drop_channels(edf_source, edf_target=None, to_keep=None, to_drop=None):
    """
    Remove channels from an edf file using pyedflib.
    Save the file as edf_target. 
    For safety reasons, no source files can be overwritten.
    
    :param edf_source: The source edf file
    :param edf_target: Where to save the file. 
                       If None, will be edf_source+'dropped.edf'
    :param to_keep: A list of channel names or indices (int) that will be kept.
                    Strings will always be interpreted as channel names.
                    'to_keep' will overwrite any droppings proposed by to_drop
    :param to_drop: A list of channel names or indices (int) that should be dropped.
                    Strings will be interpreted as channel names.
    """
    # convert to list if necessary
    if isinstance(to_keep, (int, str)): to_keep = [to_keep]
    if isinstance(to_drop, (int, str)): to_drop = [to_drop]
    
    # check all parameters are good
    assert to_keep is None or to_drop is None, 'Supply only to_keep xor to_drop'
    if to_keep is not None:
        assert all([isinstance(ch, (str, int)) for ch in to_keep]), 'channels must be int or string'
    if to_drop is not None:
        assert all([isinstance(ch, (str, int)) for ch in to_drop]), 'channels must be int or string'
    assert ospath.exists(edf_source), 'source file {} does not exist'.format(edf_source)
    assert edf_source!=edf_target, 'For safet, target must not be source file.'
        
    if edf_target is None: edf_target = ospath.splitext(edf_source)[0] + '_dropped.edf'
    if ospath.exists(edf_target): warnings.warn('Target file will be overwritten')
    
    ch_names = read_edf_header(edf_source)['channels']
    # convert to all lowercase for compatibility
    ch_names = [ch.lower() for ch in ch_names]
    ch_nrs = list(range(len(ch_names)))
    
    if to_keep is not None:
        for i,ch in enumerate(to_keep):
            if isinstance(ch,str):
                ch_idx = ch_names.index(ch.lower())
                to_keep[i] = ch_idx
        load_channels = to_keep.copy()
    elif to_drop is not None:
        for i,ch in enumerate(to_drop):
            if isinstance(ch,str):
                ch_idx = ch_names.index(ch.lower())
                to_drop[i] = ch_idx 
        to_drop = [len(ch_nrs)+ch if ch<0 else ch for ch in to_drop]

        [ch_nrs.remove(ch) for ch in to_drop]
        load_channels = ch_nrs.copy()
    else:
        raise ValueError
        
    signals, signal_headers, header = read_edf(edf_source, ch_nrs=load_channels, digital=True)
    
    write_edf(edf_target, signals, signal_headers, header, digital=True)
    return edf_target
