# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:53:08 2019

based upon https://github.com/skjerns/AutoSleepScorerDev

@author: SimonKern
"""
from difflib import SequenceMatcher
import logging as log
import tqdm as tqdm
import numpy as np
import ospath
import re
import sleep_utils

def natsort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]    
    
class SleepSet():
    
    """
    A SleepSet is a container for several Patients, where each Patient
    corresponds to one PSG recording (ie. an EDF file). This container
    makes it easier to do bulk operations, feature extraction etc.
    on a whole set of Patients    
    """
    
    def __init__(self, patient_list=[], resample=None, channel=None):
        """
        Load a list of Patients (edf format). Resample if necessary.
        
        :param patient_list: A list of strings pointing to a sleep record
        :param resample: resample all records to this frequency
        :param channel:  specify which channel to load
        """
        self.records = []
        assert type(patient_list)==list, 'patient_list must be type list'
        
        enable_progressbar = any([type(x) is str for x in patient_list])
        enable_natsort = all([type(x) is str for x in patient_list])
        
        if enable_natsort: 
            patient_list = sorted(patient_list, key=natsort_key)
        
        for patient in tqdm(patient_list, desc='[INFO] Loading Patients', 
                           disable=not enable_progressbar, leave=True):
            self.add(patient, resample=resample, channel=channel)   
        return None
    
    
    def __iter__(self):
        """
        iterate through all patients in this set
        """
        return self.patients.__iter__()
    
    
    def __getitem__(self, key):
        """
        grant access to the set with slices and indices and keys
        """
        # if it's one item: return this one item
        if type(key)==slice:
            items = self.patients.__getitem__(key)        
        elif type(key)==np.ndarray: 
            items = [self.patients.__getitem__(i) for i in key]
        elif str(type(key)())=='0': return self.patients[key]
        else:
            raise KeyError('Unknown key type:{}, {}'.format(type(key), key))
        return SleepSet(items)
    
    
    def __len__(self):
        """return the number of patients in this set"""
        return len(self.patients)  
    
    
    def add(self, patient_file, resample=None, channel=None):
        """
        Inserts a Patient to the SleepSet
        
        :param patient: Either a Patient or a file string to an ie. EDF
        :param resample: Resample the signal to this frequency
        :param channel: Load this channel explicitly
        """
        if type(patient_file) is str:
            patient = Patient(patient_file, resample=resample)
        else:
            patient = patient_file
        self.patients.append(patient)
        self.files = [patient.edf_file for sr in self]
        return self
    
    
    
class Patient():
    """
    A Patient contains the data of one edf file.
    It facilitates the automated extraction of features, loading of data
    and hypnogram as well as visualization of the record, plus statistical
    analysis.
    """
    
    def __len__(self):
        """
        returns the length of this PSG in seconds
        """
        seconds = int(len(self.data)//(self.sfreq))
        return seconds
    
    def __repr__(self):
        file = ospath.basename(self.edf_file) if self.edf_file is not None else 'None' 
        if len(file) > 25:
            file = file[:17] + '[..]' + file[-7:]
        length = int(len(self)//3600)
        s = 'SleepRecord({}, {} Hz, {} h.)'.format(file, self.sfreq, length)
        return s
    
    def __init__(self, edf_file=None, hypno_file=None, resample=None,
                 verbose=True, channel=None, ch_type='ECG'):
        """
        Loads the data of one Patient (ie. an EDF file), and its corresponding
        hypnogram annotation. The channels to be loaded can be specified.
        
        :param patient: a link to an EEG recording
        :param hypno_file: Load a specific 
                           If None, will be infered automatically
                           If False, will be ignored.
        :param resample: Resample signal after loading to this sampling frequency
        :param channel: Tells the system explicitely which channel to use from the EEG
        :returns: Patient(edf)
        """

        self.data = np.zeros(1)
        self.raw_hypno = np.zeros(1)
        self.epochs = np.empty(1) # here our extracted epochs will be stored
        self.epochlen = None # epochlen defaults to 30
        self.ch_type = ch_type

        
        self.hypno = None    # this has the converted hypnogram inside
        self.edf_file = ''

        self.sfreq = np.inf                     # sampling frequency of the data
        self.loaded_channel = channel        # which channel has been loaded will be stored here

        self.preprocessed = False          # indicates if this SleepRecord has been preprocessed
        
        if verbose==True:
            log.basicConfig(
                    format='[%(levelname)s] %(message)s',
                    level=log.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')     
        else:
            log.basicConfig(
                    format='[%(levelname)s] %(message)s',
                    level=log.CRITICAL,
                    datefmt='%Y-%m-%d %H:%M:%S')   
            
        # add a dummy windowing method that segments in 30 seconds as standard epochs
        if edf_file is not None:
            self.load(edf_file=edf_file, hypno_file=hypno_file, channel=channel)
            
        if resample is not None:
            self.resample(resample)
            

            
    
    def load(self, edf_file, hypno_file=None, channel=None):
        """
        Loads a record file and a hypno file.
        If a hypno-file is not supplied, a best fit is assumed
        
        according to the record_file name.
        :param record_file: a sleep record file
        :param hypno_file: a hypnogram annotation file
        :param hypno_pattern: which type of hypnogram to load
        """
        self.load_record(edf_file, channel=channel)
           
        # if we have no hypnogram annotation file, try to infer the file automatically
        if hypno_file is None:
            hypno_file = self._guess_hypnofile(edf_file)
            
        if hypno_file!=False:
            self.load_hypno(hypno_file)
        
        
    def load_record(self, edf_file, channel=None):
        """
        receives a link to a file, infers the file type
        and loads the data into this class into .data
        
        :param record_file: A link to a record file
        :param sfreq: Use this sampling frequency
        :param channel: Use this channel, default will be infered or is 0
        :returns: the raw array
        """
        
        if not ospath.isfile(edf_file):
            raise FileNotFoundError('edf file does not exist: {}'.format(edf_file))
        
        self.edf_file = ospath.join(edf_file)

        if not ospath.splitext(edf_file)[1] in ['.edf', '.bdf']:
            raise NotImplementedError('Only EDF/BDF/BDF+ files are supported')

        header = sleep_utils.read_edf_header(edf_file)
        channels = [ch.upper() for ch in header['channels']]
        if isinstance(channel, int): 
            ch_idx = channel
        else :
            try:
                ch_idx = channels.index(channel.upper())
                log.info('Loading Channel #{}: {}'.format(channel, channels[ch_idx]))
            except:
                ch_idx = self._guess_channel_name(channels)
                
        self.loaded_channel = channels[ch_idx]
        self.sfreq = header['SignalHeaders'][ch_idx]['sample_rate']
        
        data, _, header = sleep_utils.read_edf(edf_file, digital=False, ch_nrs=ch_idx, verbose=False)
        data = data.squeeze()
        
        self.data = data
        self.header = header
        self.startdate = header['startdate']
        self.starttime = (self.startdate.hour * 60 + self.startdate.minute) * 60 + self.startdate.second
                    
        self.preprocessed = False
      
        hours = len(self.data)//self.sfreq/60/60
        log.info('Loaded {:.1f} hours of {} with sfreq={}'.format(hours, self.ch_type, self.sfreq))
        
        return self
    
    
    def load_hypno(self, hypno_file, epochlen_infile=None):
        """
        Loads a hypnogram and stores results in self.hypno
        Can parse both hypnogram types created by VisBrain
        Internally, we store for each second one annotation
        For preprocessing, one annotation per segment is used.
        The concurrent transformation during processing is important
        
        :param hypno_file: a path to the hypnogram
        :param epochlen_infile: how many seconds per label in original file
        """
        self.hypno_file = hypno_file
        sfreq = self.sfreq
        exp_seconds = None if self.data.size<=1 else len(self.data)//sfreq
        print(self.data.size>1 )
        print(exp_seconds)
        hypno = sleep_utils.read_hypnogram(hypno_file, epochlen_infile=epochlen_infile, 
                                 exp_seconds=exp_seconds)
        log.debug('loaded hypnogram {}'.format(hypno_file))
        self.raw_hypno = hypno
        self.hypno = hypno.copy()
        
        
    def resample(self, target_sfreq):
        """
        Uses a temporary mne array to resample self.data using MNE
        MNE is used as it offers a resampling function that is 
        optimized for EEG/electrophysiological data.
        
        :param target_sfreq: the target sampling frequency
        """    
        import mne
        if np.round(self.sfreq)!=target_sfreq:
            data = self.data
            info = mne.create_info(ch_names=['eeg'], sfreq=self.sfreq, ch_types=['eeg'])
            raw_mne = mne.io.RawArray(data, info, verbose='ERROR')
            resampled = raw_mne.resample(target_sfreq, n_jobs=3)
            new_data = resampled.get_data().squeeze()
            self.data = new_data
        else:
            log.debug('Signal is already in {} Hz'.format(target_sfreq))        
        return self
    
    
    def _guess_channel_name(self, channels=None):
        """
        Try to guess an appropriate EEG channel to load from the file.
        Will choose in the following order of fit: F4/F3/FP1/Fp2/C3/C4/ any EEG
        
        :param channels: a list of channel names
        :returns: argpos of the best fitting channel name
                  returns 0 in case of failure
        """
        if self.ch_type == 'EEG':
            keywords = ['F4', 'F3', 'FP1', 'FP2', 'C3' , 'C4', 'EEG']
        elif self.ch_type == 'ECG':
            keywords = ['ECG', 'EKG']
        else:
            log.error('Invalid ch_type, select from [ECG, EEG]')

        channels = [ch.upper() for ch in channels]
        for k in keywords:
            matches = [k in ch for ch in channels]
            if any(matches):
                ch_idx = np.argmax(matches)
                ch_name = channels[ch_idx].upper()
                log.info('Infering channel # {}: {} from {}...'.format(ch_idx, ch_name, channels[:4]))
                return ch_idx
        log.warning('No channel matches in {}. Taking channel 0={}'.format(channels, channels[0]))
        return 0  
    
    
    @staticmethod
    def _guess_hypnofile(edf_file):
        """
        given an edf_file, tries to find a matching hypnogram file.
        the pattern that is looked for is RECORDNAME + [.csv or .dat or .txt]
        
        :param edf_file: A string linking a edf file
        """
        
        def str_match(s1, s2):
            return SequenceMatcher(None, s1, s2).ratio()
                
        folder = ospath.dirname(edf_file)
        rfile  = ospath.splitext(ospath.basename(edf_file))[0]
        
        files  = ospath.list_files(folder, exts=['txt', 'csv', 'dat'], relative=True) 
        # filter out every RECORDNAME.*
        matching = [file for file in files if 
                    (file.lower().startswith(rfile.lower()) 
                    and len(rfile)<len(file))]
        if len(matching)==1:
            hyp = ospath.join(folder, matching[0])

        elif len(matching)>1:
            hyp = matching[0]
            log.warning('Several hypnograms found: '.format(matching))
        else:
            log.warning('No matching hypnogram for {} '.format(ospath.basename(edf_file)))
            return False
        log.info('hypnogram: Matched {} to {}'.format(ospath.basename(hyp), 
                                                      ospath.basename(edf_file)))
        return hyp
            