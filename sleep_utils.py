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
from datetime import datetime
import h5py

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
    conv_dict = {'Wake':0, 'N1': 1, 'N2': 2, 'N3': 3, 'N4':3, 'REM': 4, 'Art': 5,
                 0:0, 1:1, 2:2, 3:3, 4:4, -1:5, 5:5}
    
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
            if epochlen_infile!=sec_diff: 
                warnings.warn('Epochlen in file is {} but {} would be selected'.format(sec_diff, epochlen_infile))
            
            stage = conv_dict[stage]
            stages.extend([stage]*sec_diff)
    
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
            s = conv_dict[s]
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
        stages = np.array([conv_dict[l] for l in np.array(lines).flatten()])
    
    # for the Dreams Database 
    # http://www.tcts.fpms.ac.be/~devuyst/Databases/DatabaseSubjects/    
    elif mode=='dreams':
        epochlen_infile = 5
        conv_dict = {-2:5,-1:5, 0:5, 1:3, 2:2, 3:1, 4:4, 5:0}    
        lines = [[int(line)] for line in lines[1:] if len(line)>0]
        lines = [[line]*epochlen_infile for line in lines]
        stages = np.array([conv_dict[l] for l in np.array(lines).flatten()])
        
    # for hypnogram created with Alice 5 software
    elif mode=='alice':
        epochlen_infile = 30
        conv_dict = {'WK':0,'N1':1, 'N2':2, 'N3':3, 'REM':4}  
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



def h5py2dict(fname):
    hdf5 = h5py.File(fname, 'r')
    def unpack(hdf5, depth=0):
        if depth==99:raise Exception
        if isinstance(hdf5, (h5py._hl.group.Group)):
            d = {}
            for key in hdf5:
                elem   = hdf5[key]
                d[key] = unpack(elem, depth=depth+1)
            return d
        elif isinstance(hdf5, h5py._hl.dataset.Dataset):
            if not 'MATLAB_class' in hdf5.attrs: return None
            mtype = hdf5.attrs['MATLAB_class'].decode()
            npdtypes = [x for x in np.__dict__ if isinstance(np.__dict__[x],type)]
            if mtype in npdtypes:
                dtype = np.__dict__[mtype]
                arr = np.array(hdf5, dtype=dtype).T
                return arr.squeeze() if arr.size==1 else arr
            if mtype=='char': return ''.join([chr(x) for x in hdf5])
            if mtype=='logical': 
                arr = np.array(hdf5, dtype=bool)
                return arr.squeeze() if arr.size==1 else arr
            if mtype=='canonical empty': return 'None'
            if mtype=='cell':
                cell = []
                for ref in hdf5:
                    for r in ref:
                        entry = unpack(refs[r])
                        cell.append(entry)
                return cell
            print('Unknown data type ', mtype)
            return 'Unknown type {}'.format(mtype)
        
    if '#refs#' in hdf5:
        refs = hdf5['#refs#']
        
    d = {}
    for var in hdf5:
        if var=='#refs#':continue
        d[var] = unpack(hdf5[var])
    hdf5.close()
    return d
   