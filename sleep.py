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
import matplotlib.pyplot as plt
from unisens import Unisens, CustomEntry

def natsort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]    
    
class SleepSet():
    """
    A SleepSet is a container for several Patients, where each Patient
    corresponds to one PSG recording (ie. an EDF file). This container
    makes it easier to do bulk operations, feature extraction etc.
    on a whole set of Patients.
    """
    
    def __init__(self, patient_list=None, patients=None):
        """
        Load a list of Patients (edf format). Resample if necessary.
        
        :param patient_list: A list of strings pointing to a sleep record
        :param resample: resample all records to this frequency
        :param channel:  specify which channel to load
        """
        assert isinstance(patient_list, list), 'patient_list must be type list'
        self.patients = []
        
        # must be either Patients or strings to make patients from.
        all_patients = all([isinstance(x, Patient) for x in patient_list])
        all_strings = all([isinstance(x, str) for x in patient_list])
        assert all_patients or all_strings, \
            "patient_list must be either strings or Patients"
        
        if all_strings: # natural sorting of file list
            patient_list = sorted(patient_list, key=natsort_key)
        
        for patient in patient_list:
            patient = Patient(patient)
            self.add(patient)   
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
    
    
    def filter(self, function):
        return SleepSet(list(filter(function, self.patients)))
    
    
    def add(self, patient):
        """
        Inserts a Patient to the SleepSet
        
        :param patient: Either a Patient or a file string to an ie. EDF
        :param resample: Resample the signal to this frequency
        :param channel: Load this channel explicitly
        """
        if isinstance(patient, Patient):
            self.patients.append(patient)
        else:
            raise ValueError(f'patient must be or Patient, is {type(patient)}')
        return self
    
    
    
class Patient(Unisens):
    """
    A Patient contains the data of one unisens data structure.
    It facilitates the automated extraction of features, loading of data
    and hypnogram as well as visualization of the record, plus statistical
    analysis. Many of its subfunctions are inherited from Unisens
    """
    def __new__(cls, folder, *args, **kwargs):
        """
        If this patient is initialized with a Patient, just return this Patient
        """
        if isinstance(folder, Patient): return folder
        return object.__new__(cls)

    def __repr__(self):
        return 'Patient'
    
    def __str__(self):
        return 'Patient'
    
    def __init__(self, folder, *args, **kwargs):
        if isinstance(folder, Patient): return None
        super().__init__(folder, *args, autosave=True, **kwargs)

    def __len__(self):
        """
        returns the length of this PSG in seconds
        """
        seconds = int(len(self.data)//(self.sfreq))
        return seconds
        
    def get_hypno(self):
        return self['hypnogram.csv'].get_data()
     
    def get_ecg(self):
        return self['ecg.csv'].get_data()

    def get_eeg(self):
        return self['eeg.csv'].get_data()
    
    def plot(self, channel='eeg', ax=None, make_new=False):
        
        if ax is None:
            plt.figure()
            ax = plt.subplot()
            
        file = f'plot_{channel}.jpg'
        
        if file in self.entries and not make_new:
            spec = self.entries[file].get_data()
            sfreq = self.entries[f'{channel}.bin'].samplingRate
            plt.imshow(spec)
        else:
            signal = self.entries[f'{channel}.bin'].get_data()[0]
            sfreq = self.entries[f'{channel}.bin'].samplingRate
            spec = sleep_utils.specgram_multitaper(signal, int(sfreq), 
                                                   show_plot=False)
            plt.imshow(spec)
            
            CustomEntry(file, parent=self).set_data(spec)
        plt.title(f'{channel}, {sfreq} Hz')