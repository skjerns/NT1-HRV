# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:19:26 2019

@author: skjerns
"""
from pyedflib.highlevel import *
import os
import gc
import warnings
import ospath #pip install https://github.com/skjerns/skjerns-utils
import numpy as np
import pyedflib #pip install https://github.com/skjerns/pyedflib/archive/custom_version.zip
import time
from tqdm import tqdm
from datetime import datetime
import dateparser
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
import matplotlib



def read_hypnogram(hypno_file, epochlen = 30, epochlen_infile=None, mode='auto', exp_seconds=None):
    """
    reads a hypnogram file as created by VisBrain or as CSV type 
    
    :param hypno_file: a path to the hypnogram
    :param epochlen: how many seconds per label in output
    :param epochlen_infile: how many seconds per label in original file
    :param mode: 'auto', 'time' or 'csv', see SleepDev/docs/hypnogram.md
    :param exp_seconds: How many seconds does the matching recording have?
    """
    assert str(type(epochlen)()) == '0'
    assert epochlen_infile is None or str(type(epochlen_infile)()) == '0'

    with open(hypno_file, 'r') as file:
        content = file.read()
        content = content.replace('\r', '') # remove windows style \r\n
        
    #conversion dictionary
    conv_dict = {'WAKE':0, 'WACH':0, 'WK':0,  'N1': 1, 'N2': 2, 'N3': 3, 'N4':3, 'REM': 4,
                 0:0, 1:1, 2:2, 3:3, 4:4, -1:5, 5:5, 'ART': 5, 'A':5, 'ARTEFAKT':5, 'MT':5}
    
    lines = content.split('\n')
    if mode=='auto':
        if lines[0].startswith('*'): # if there is a star, we assume it's the visbrain type
            mode = 'visbrain'
        elif lines[0].replace('-', '').isnumeric():
            mode = 'csv'
        elif lines[0].startswith('[HypnogramAASM]'):
            mode = 'dreams'
        elif lines[0].startswith(' Epoch Number ,Start Time ,Sleep Stage'):
            mode = 'alice'
        elif 'abstime' in lines[0]:
            mode = 'dat'
        elif lines[0].startswith('Signal ID:'):
            mode = 'somnoscreen'
        else :
            mode==None
    
    # reading file in format as used by koden
    if mode=='dat':
        if epochlen_infile is not None:
            warnings.warn('epochlen_infile has been supplied, but hypnogram is' 
                          'time based, will be ignored')
        elif exp_seconds and not epochlen_infile:
            epochlen_infile=exp_seconds//len(lines)
            print('[INFO] Assuming csv annotations with one entry per {} seconds'.format(epochlen_infile))

        stages = []
        for line1, line2 in zip(lines[1:-1], lines[2:]):
            if len(line1.strip())==0: continue # skip empty lines
            if len(line2.strip())==0: continue # skip empty lines

            curr_t, _, stage, *_ = line1.split('\t')
            next_t,*_ = line2.split('\t')
            curr_t = datetime.strptime(curr_t, '%Y-%m-%d %H:%M:%S')
            next_t = datetime.strptime(next_t, '%Y-%m-%d %H:%M:%S')
            assert next_t > curr_t, 'timestamp 2 is smaller than 1? {} < {}'.format(next_t, curr_t)
            
            sec_diff = (next_t - curr_t).seconds
            if exp_seconds and epochlen_infile!=sec_diff: 
                warnings.warn('Epochlen in file is {} but {} would be selected'.format(sec_diff, epochlen_infile))
            
            stage = conv_dict[stage.upper()]
            stages.extend([stage]*sec_diff)
    
    elif mode=='somnoscreen':
        if epochlen_infile is not None:
            warnings.warn('epochlen_infile has been supplied, but information is in file, will be ignored')
        
        epochlen_infile = int(lines[5].replace('Rate: ', '').replace('s',''))
        stages = []
        for line in lines[6:]:
            if len(line.strip())==0: continue # skip empty lines
            
            _,stage = line.split('; ')
            stage = conv_dict[stage.upper()]
            stages.extend([stage]*epochlen_infile)
            
            
            
    # read hypnogram as written by visbrain (time based)
    elif mode=='visbrain':
        if epochlen_infile is not None:
            warnings.warn('epochlen_infile has been supplied, but hypnogram is time based,'
                          'will be ignored')
        stages = []
        prev_t = 0
        for line in lines:
            if len(line.strip())==0:   continue
            if line[0] in '*#%/\\"\'': continue # this line seems to be a comment
            s, t = line.split('\t')
            t = float(t)
            s = conv_dict[s.upper()]
            l = int(np.round((t-prev_t))) # length of this stage
            stages.extend([s]*l)
            prev_t = t
            
    # read hypnogram as simple CSV file       
    elif mode=='csv':
        if exp_seconds and not epochlen_infile:
            epochlen_infile=exp_seconds//len(lines)
            print('[INFO] Assuming csv annotations with one entry per {} seconds'.format(epochlen_infile))

        elif epochlen_infile is None: 
            if len(lines) < 2400: # we assume no recording is longer than 20 hours
                epochlen_infile = 30
                print('[INFO] Assuming csv annotations are per epoch')
            else:
                epochlen_infile = 1
                print('[INFO] Assuming csv annotations are per second')
        lines = [[int(line)] for line in lines if len(line)>0]
        lines = [[line]*epochlen_infile for line in lines]
        stages = np.array([conv_dict[l.upper()] for l in np.array(lines).flatten()])
    
    # for the Dreams Database 
    # http://www.tcts.fpms.ac.be/~devuyst/Databases/DatabaseSubjects/    
    elif mode=='dreams':
        epochlen_infile = 5
        conv_dict = {-2:5,-1:5, 0:5, 1:3, 2:2, 3:1, 4:4, 5:0}    
        lines = [[int(line)] for line in lines[1:] if len(line)>0]
        lines = [[line]*epochlen_infile for line in lines]
        stages = np.array([conv_dict[l.upper()] for l in np.array(lines).flatten()])
        
    # for hypnogram created with Alice 5 software
    elif mode=='alice':
        epochlen_infile = 30
        lines = [line.split(',')[-1] for line in lines[1:] if len(line)>0]
        lines = [[line]*epochlen_infile for line in lines]
        try: stages = np.array([conv_dict[l] for l in np.array(lines).flatten()])
        except KeyError as e:
            print('Unknown sleep stage in file')
            raise e
    else:
        raise ValueError('This is not a recognized hypnogram: {}'.format(hypno_file))
        
    stages = stages[::epochlen]
    if len(stages)==0:
        print('[WARNING] hypnogram loading failed, len == 0')
    return np.array(stages)



def infer_eeg_channels(ch_names):
    """
    This function receives a list of channel names and will return
    one frontal, one central and one occipital channel.    
    """
    
    f = ['EEG Fz', 'EEG F4', 'EEG Fpz', 'EEG Fp1', 'EEG Fp2']
    c = ['EEG C4', 'EEG C3']
    o = ['EEG Oz', 'EEG O2', 'EEG O1']
    
    found = []

    # find frontal channel
    for ch in ch_names:
        if any([x in ch for x in f]):
            found.append(ch)
            break
    # find central channel
    for ch in ch_names:
        if any([x in ch for x in c]):
            found.append(ch)
            break
    # find occipital channel
    for ch in ch_names:
        if any([x in ch for x in o]):
            found.append(ch)
            break
    return found
    
def infer_eog_channels(ch_names):
    """
    This function receives a list of channel names and will return
    one frontal, one central and one occipital channel.    
    """
    
    eog = ['EOG ROC', 'EOG LOC']
    found = []

    # find frontal channel
    for ch in ch_names:
        if any([x in ch for x in eog]):
            found.append(ch)
    return found

def infer_emg_channels(ch_names):
    """
    This function receives a list of channel names and will return
    one frontal, one central and one occipital channel.    
    """
    emg = ['EMG Chin']
    found = []

    # find frontal channel
    for ch in ch_names:
        if any([x in ch for x in emg]):
            found.append(ch)
    return found


def hypno2time(hypno, seconds_per_epoch=1):
    """
    Converts a hypnogram based in epochs into the format as defined
    by VisBrain: http://visbrain.org/sleep.html#save-hypnogram
    """
    hypno = np.repeat(hypno, seconds_per_epoch)
    s = '*Duration_sec {}\n'.format(len(hypno))
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Art']
    d = dict(enumerate(stages))
    hypno_str = [d[h] for h in hypno]
    
    last_stage=hypno_str[0]
    
    for second, stage in enumerate(hypno_str):
        if stage!=last_stage:
            s += '{}\t{}\n'.format(last_stage, second)
            last_stage=stage
    s += '{}\t{}\n'.format(stage, second+1)
    return s


def write_hypnogram(hypno, filename, seconds_per_annotation=30, 
                     comment=None, overwrite=False):
    """
    Save a hypnogram based on annotations per epochs in VisBrain style
    (ie. The exact onset of each sleep stage is annotated in time space.)
    This format is recommended for saving hypnograms as it avoids ambiguity.
    
    :param filename: where to save the data
    :param hypno: The hypnogram either as list or np.array
    :param seconds_per_epoch: How many seconds each annotation contains
    :param comment: Add a comment to the beginning of the file
    :param overwrite: overwrite file?
    """
    assert not ospath.exists(filename) or overwrite,  \
              'File already exists, no overwrite'
    hypno = np.repeat(hypno, seconds_per_annotation)
    hypno_str = hypno2time(hypno)
    if comment is not None:
        comment = comment.replace('\n', '\n*')
        hypno_str = '*' + comment + '\n' + hypno_str
        hypno_str = hypno_str.replace('\n\n', '\n')
    with open(filename, 'w') as f:
        f.write(hypno_str)    
    return True

def minmax2lsb(dmin, dmax, pmin, pmax):
    """
    converts the edf min/max values to lsb and offset (x*m+b)
    """
    lsb = (pmax - pmin) / (dmax - dmin)
    offset = pmax / lsb - dmax
    return lsb, offset

def make_header(technician='', recording_additional='', patientname='',
                patient_additional='', patientcode= '', equipment= '',
                admincode= '', gender= '', startdate=None, birthdate= ''):
    """
    A convenience function to create an EDF header (a dictionary) that
    can be used by pyedflib to update the main header of the EDF
    """
    if not( startdate is None or isinstance(startdate, datetime)):
        warnings.warn('must be datetime or None, is {}: {},attempting convert'\
                      .format(type(startdate), startdate))
        startdate = dateparser.parse(startdate)
    if not (birthdate == '' or isinstance(birthdate, (datetime,str))):
        warnings.warn('must be datetime or empty, is {}, {}'\
                      .format(type(birthdate), birthdate))
        birthdate = dateparser.parse(birthdate)
    if startdate is None: 
        now = datetime.now()
        startdate = datetime(now.year, now.month, now.day, 
                             now.hour, now.minute, now.second)
        del now
    if isinstance(birthdate, datetime): 
        birthdate = birthdate.strftime('%d %b %Y')
    local = locals()
    header = {}
    for var in local:
        if isinstance(local[var], datetime):
            header[var] = local[var]
        else:
            header[var] = str(local[var])
    return header

def make_signal_header(label, dimension='uV', sample_rate=256, 
                       physical_min=-200, physical_max=200, digital_min=-32768,
                       digital_max=32767, transducer='', prefiler=''):
    """
    A convenience function that creates a signal header for a given signal.
    This can be used to create a list of signal headers that is used by 
    pyedflib to create an edf. With this, different sampling frequencies 
    can be indicated.
    
    :param label: the name of the channel
    """
    signal_header = {'label': label, 
               'dimension': dimension, 
               'sample_rate': sample_rate, 
               'physical_min': physical_min, 
               'physical_max': physical_max, 
               'digital_min':  digital_min, 
               'digital_max':  digital_max, 
               'transducer': transducer, 
               'prefilter': prefiler}
    return signal_header


def make_signal_headers(list_of_labels, dimension='uV', sample_rate=256, 
                       physical_min=-200, physical_max=200, digital_min=-32768,
                       digital_max=32767, transducer='', prefiler=''):
    """
    A function that creates signal headers for a given list of channel labels.
    This can only be used if each channel has the same sampling frequency
    
    :param list_of_labels: A list with labels for each channel.
    :returns: A dictionary that can be used by pyedflib to update the header
    """
    signal_headers = []
    for label in list_of_labels:
        header = make_signal_header(label, dimension=dimension, sample_rate=sample_rate, 
                                    physical_min=physical_min, physical_max=physical_max,
                                    digital_min=digital_min, digital_max=digital_max,
                                    transducer=transducer, prefiler=prefiler)
        signal_headers.append(header)
    return signal_headers



def write_edf(edf_file, signals, signal_headers, header, digital=False,
              correct=False):
    """
    Write signals to an edf_file. Header can be generated on the fly.
    
    :param signals: The signals as a list of arrays or a ndarray
    :param signal_headers: a list with one signal header(dict) for each signal.
                           See pyedflib.EdfWriter.setSignalHeader
    :param header: a main header (dict) for the EDF file, see 
                   pyedflib.EdfWriter.setHeader for details
    :param digital: whether signals are presented digitally 
                    or in physical values
    
    :returns: True if successful, False if failed
    """
    assert header is None or isinstance(header, dict), \
        'header must be dictioniary'
    assert isinstance(signal_headers, list), \
        'signal headers must be list'
    assert len(signal_headers)==len(signals), \
        'signals and signal_headers must be same length'

    n_channels = len(signals)
    
    # check min and max values
    if digital==True and correct:
        for sig, sigh in zip(signals,signal_headers):
            dmin, dmax = sigh['digital_min'], sigh['digital_max']
            pmin, pmax = sigh['physical_min'], sigh['physical_max']
            ch_name=sigh['label']
            if dmin>dmax: 
                 logging.warning('{}: dmin>dmax, {}>{}, will correct'.format(\
                                 ch_name, dmin, dmax))
                 dmin, dmax = dmax, dmin
                 sig *= -1
            if pmin>pmax: 
                 logging.warning('{}: pmin>pmax, {}>{}, will correct'.format(\
                                  ch_name, pmin, pmax))
                 pmin, pmax = pmax, pmin
                 sig *= -1
            dsmin, dsmax = round(sig.min()), round(sig.max())
            psmin = dig2phys(dsmin, dmin, dmax, pmin, pmax)
            psmax = dig2phys(dsmax, dmin, dmax, pmin, pmax)
            min_dist = np.abs(dig2phys(1, dmin, dmax, pmin, pmax))
            if dsmin<dmin:
                logging.warning('{}:Digital signal minimum is {}'\
                                ', but value range is {}, will correct'.format\
                                (ch_name, dmin, dsmin))
                sigh['digital_min'] = dsmin
            if dsmax>dmax:
                logging.warning('{}:Digital signal maximum is {}'\
                                ', but value range is {}, will correct'.format\
                                (ch_name, dmax, dsmax))
                # sigh['digital_max'] = dsmax
            if psmax-min_dist>pmax:
                logging.warning('{}:Phyiscal signal maximum is {}'\
                                ', but value range is {}, will correct'.format\
                                (ch_name, pmax, psmax))
                sigh['physical_max'] = psmax
            if psmin+min_dist<pmin:
                logging.warning('{}:Physical signal minimum is {}'\
                                ', but value range is {}, will correct'.format\
                                (ch_name, pmin, psmin))
                sigh['physical_min'] = psmin
                
    # also add annotations
    annotations = header.get('annotations', '')
         
    with pyedflib.EdfWriter(edf_file, n_channels=n_channels) as f:  
        f.setSignalHeaders(signal_headers)
        f.setHeader(header)
        f.writeSamples(signals, digital=digital)
        for annotation in annotations:
            f.writeAnnotation(*annotation)
    del f
    return os.path.isfile(edf_file) 


def compare_edf(edf_file1, edf_file2, verbose=True, threading=True):
    """
    Loads two edf files and checks whether the values contained in 
    them are the same. Does not check the header data
    """
    if verbose: print('verifying data')
    files = [(edf_file1, True), (edf_file2, True)]
    # di
    backend = 'loky' if threading else 'sequential'
    results = Parallel(n_jobs=2, backend=backend)(delayed(read_edf)\
             (file, digital=digital, verbose=False) for file, \
             digital in tqdm(files, disable=not verbose))  

    signals1, signal_headers1, _ =  results[0]
    signals2, signal_headers2, _ =  results[1]

    for i, sigs in enumerate(zip(signals1, signals2)):
        s1, s2 = sigs
        s1 = np.abs(s1)
        s2 = np.abs(s2)
        assert np.allclose(s1, s2), 'Error, digital values of {}'\
            ' and {} for ch {}: {} are not the same'.format(
                edf_file1, edf_file2, signal_headers1[i]['label'], 
                signal_headers2[i]['label'])
        gc.collect()

    dmin1, dmax1 = signal_headers1[i]['digital_min'], signal_headers1[i]['digital_max']
    pmin1, pmax1 = signal_headers1[i]['physical_min'], signal_headers1[i]['physical_max']
    dmin2, dmax2 = signal_headers2[i]['digital_min'], signal_headers2[i]['digital_max']
    pmin2, pmax2 = signal_headers2[i]['physical_min'], signal_headers2[i]['physical_max']

    for i, sigs in enumerate(zip(signals1, signals2)):
        s1, s2 = sigs
     
        # convert to physical values, no need to load all data again
        s1 = dig2phys(s1, dmin1, dmax1, pmin1, pmax1)
        s2 = dig2phys(s2, dmin2, dmax2, pmin2, pmax2)
        
        # now we can remove the signals from the list to save memory
        signals1[i] = None
        signals2[i] = None
        
        # compare absolutes in case of inverted signals
        s1 = np.abs(s1)
        s2 = np.abs(s2)
        
        min_dist = np.abs(dig2phys(1, dmin1, dmax1, pmin1, pmax1))
        close =  np.mean(np.isclose(s1, s2, atol=min_dist))
        assert close>0.99, 'Error, physical values of {}'\
            ' and {} for ch {}: {} are not the same: {:.3f}'.format(
                edf_file1, edf_file2, signal_headers1[i]['label'], 
                signal_headers2[i]['label'], close)
        gc.collect()
    return True


def change_polarity(edf_file, channels, new_file=None):
    if new_file is None: 
        new_file = os.path.splitext(edf_file)[0] + '_inv.edf'
    
    if isinstance(channels, str): channels=[channels]
    channels = [c.lower() for c in channels]

    signals, signal_headers, header = read_edf(edf_file, digital=True)
    for i,sig in enumerate(signals):
        shead = signal_headers[i]
        label = signal_headers[i]['label'].lower()
        if label in channels:
            print('inverting {}'.format(label))
            shead['physical_min']*=-1
            shead['physical_max']*=-1
    write_edf(new_file, signals, signal_headers, header, digital=True)
    compare_edf(edf_file, new_file)
   
    
    
def specgram_multitaper(data, sfreq, sperseg=30, perc_overlap=1/3,
                        lfreq=0, ufreq=40, show_plot=True, ax=None):
    """
    Display EEG spectogram using a multitaper from 0-30 Hz

    :param data: the data to visualize, should be of rank 1
    :param sfreq: the sampling frequency of the data
    :param sperseg: number of seconds to use per FFT
    :param noverlap: percentage of overlap between segments
    :param lfreq: Lower frequency to display
    :param ufreq: Upper frequency to display
    :param show_plot: If false, only the mesh is returned, but not Figure opened
    :param ax: An axis where to plot. Else will create a new Figure
    :returns: the resulting mesh as it would be plotted
    """
    
    if ax is None:
        plt.figure()
        ax=plt.subplot(1,1,1)
        
    assert isinstance(show_plot, bool), 'show_plot must be boolean'
    nperseg = int(round(sperseg * sfreq))
    overlap = int(round(perc_overlap * nperseg))

    f_range = [lfreq, ufreq]

    freq, xy, mesh = spectrogram_lspopt(data, sfreq, nperseg=nperseg,
                                       noverlap=overlap, c_parameter=20.)
    if mesh.ndim==3: mesh = mesh.squeeze().T
    mesh = 20 * np.log10(mesh+0.0000001)
    idx_notfinite = np.isfinite(mesh)==False
    mesh[idx_notfinite] = np.min(mesh[~idx_notfinite])

    f_range[1] = np.abs(freq - ufreq).argmin()
    sls = slice(f_range[0], f_range[1] + 1)
    freq = freq[sls]

    mesh = mesh[sls, :]
    mesh = mesh - mesh.min()
    mesh = mesh / mesh.max()
    if show_plot:
        ax.imshow(np.flipud(mesh), aspect='auto')
        formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(int(s*(sperseg-overlap/sfreq)))))
        ax.xaxis.set_major_formatter(formatter)
        if xy[-1]<3600*7: # 7 hours is half hourly
            tick_distance = max(np.argmax(xy>sperseg*60),5) #plot per half hour
        else: # more than 7 hours hourly ticks
            tick_distance = np.argmax(xy>sperseg*60)*2 #plot per half hour
        two_hz_pos = np.argmax(freq>1.99999999)
        ytick_pos = np.arange(0, len(freq), two_hz_pos)
        ax.set_xticks(np.arange(0, mesh.shape[1], tick_distance))
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(np.arange(ufreq, lfreq-1, -2))
        ax.set_xlabel('Time after onset')
        ax.set_ylabel('Frequency')
        warnings.filterwarnings("ignore", message='This figure includes Axes that are not compatible')
        plt.tight_layout()
    return mesh


def plot_hypnogram(stages, labeldict=None, title=None, epochlen=30, ax=None,
                   verbose=True, xlabel=True, ylabel=True, **kwargs,):
    """
    Plot a hypnogram, the flexible way.

    A labeldict should give a mapping which integer belongs to which class
    E.g labeldict = {0: 'Wake', 4:'REM', 1:'S1', 2:'S2', 3:'SWS'}
    or {0:'Wake', 1:'Sleep', 2:'Sleep', 3:'Sleep', 4:'Sleep', 5:'Artefact'}

    The order of the labels on the plot will be determined by the order of the dictionary.

    E.g.  {0:'Wake', 1:'REM', 2:'NREM'}  will plot Wake on top, then REM, then NREM
    while {0:'Wake', 2:'NREM', 1:'NREM'} will plot Wake on top, then NREM, then REM

    This dictionary can be infered automatically from the numbers that are present
    in the hypnogram but this functionality does not cover all cases.

    :param stages: An array with different stages annotated as integers
    :param labeldict: An enumeration of labels that correspond to the integers of stages
    :param title: Title of the window
    :param epochlen: How many seconds is one epoch in this annotation
    :param ax: the axis in which we plot
    :param verbose: Print stuff or not.
    :param xlabel: Display xlabel ('Time after record start')
    :param ylabel: Display ylabel ('Sleep Stage')
    :param kwargs: additional arguments passed to plt.plot(), e.g. c='red'
    """


    if labeldict is None:
        if np.max(stages)==1 and np.min(stages)==0:
            labeldict = {0:'W', 1:'S'}
        elif np.max(stages)==2 and np.min(stages)==0:
            labeldict = {0:'W', 2:'REM', 1:'NREM'}
        elif np.max(stages)==4 and np.min(stages)==0:
            if 1 in stages:
                labeldict = {0:'W', 4:'REM', 1:'S1', 2:'S2', 3:'SWS', }
            else:
                labeldict = {0:'W', 4:'REM', 2:'S2', 3:'SWS'}
        else:
            if verbose: print('could not detect labels?')
            if 1 in stages:
                labeldict = {0:'W', 4:'REM', 1:'S1', 2:'S2', 3:'SWS', 5:'A'}
            else:
                labeldict = {0:'W', 4:'REM', 2:'S2', 3:'SWS', 5:'A'}
        if -1 in stages:
            labeldict['ARTEFACT'] = -1
        if verbose: print('Assuming {}'.format(labeldict))

    # check if all stages that are in the hypnogram have a corresponding label in the dict
    for stage in np.unique(stages):
        if not stage in labeldict:
            print('WARNING: {} is in stages, but not in labeldict, stage will be ??'.format(stage))

    # create the label order
    labels = [labeldict[l] for l in labeldict]
    labels = sorted(set(labels), key=labels.index)

    # we iterate through the stages and fetch the label for this stage
    # then we append the position on the plot of this stage via the labels-dict
    x = []
    y = []
    rem_start = []
    rem_end   = []
    for i in np.arange(len(stages)):
        s = stages[i]
        label = labeldict.get(s)
        if label is None:
            p = 99
            if '??' not in labels: labels.append('??')
        else :
            p = -labels.index(label)
        
        # make some red line markers for REM, mark beginning and end of REM
        if 'REM' in labels:
            if label=='REM' and len(rem_start)==len(rem_end):
                    rem_start.append(i-2)
            elif label!='REM' and len(rem_start)>len(rem_end):
                rem_end.append(i-1)
        if label=='REM' and i==len(stages)-1:
           rem_end.append(i+1)
            
        if i!=0:
            y.append(p)
            x.append(i-1)
        y.append(p)
        x.append(i)
    
    assert len(rem_start)==len(rem_end), 'Something went wrong in REM length calculation'

    x = np.array(x)*epochlen
    y = np.array(y)
    y[y==99] = y.min()-1 # make sure Unknown stage is plotted below all else

    if ax is None:
        plt.figure()
        ax = plt.gca()
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
    
    ax.plot(x,y, **kwargs)
    ax.set_xlim(0, x[-1])
    ax.xaxis.set_major_formatter(formatter)
    
    ax.set_yticks(np.arange(len(np.unique(labels)))*-1)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(0,x[-1],3600))
    if xlabel: plt.xlabel('Time after recording start')
    if ylabel: plt.ylabel('Sleep Stage')
    if title is not None:
        plt.title(title)

    try:
        warnings.filterwarnings("ignore", message='This figure includes Axes that are not compatible')
        plt.tight_layout()
    except Exception: pass

    # plot REM in RED here
    for start, end in zip(rem_start, rem_end):
        height = -labels.index('REM')
        ax.hlines(height, start*epochlen, end*epochlen, color='r',
                   linewidth=4, zorder=99)



    