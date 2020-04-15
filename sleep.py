# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:53:08 2019

todo: check length of hypno and features

@author: skjerns
"""
import misc
import config
import logging as log
import tqdm as tqdm
import numpy as np
import ospath
import os
import re
from tqdm import tqdm
import time
import sleep_utils
import matplotlib.pyplot as plt

from unisens import Unisens, CustomEntry, SignalEntry, EventEntry, ValuesEntry

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
    
    def __init__(self, patient_list:list=None):
        """
        Load a list of Patients (edf format). Resample if necessary.
        
        :param patient_list: A list of strings pointing to Patients
        """
        if isinstance(patient_list, str):
            patient_list = ospath.list_folders(patient_list)
        assert isinstance(patient_list, list), 'patient_list must be type list'
        self.patients = []
        
        # must be either Patients or strings to make patients from.
        all_patients = all([isinstance(x, Patient) for x in patient_list])
        all_strings = all([isinstance(x, str) for x in patient_list])
        assert all_patients or all_strings, \
            "patient_list must be either strings or Patients"
        
        if all_strings: # natural sorting of file list
            patient_list = sorted(patient_list, key=natsort_key)
            
        tqdm_loop = tqdm if all_strings else lambda x, *args,**kwargs: x
        
        for patient in tqdm_loop(patient_list, desc='Loading Patients'):
            try:
                patient = Patient(patient)
                self.add(patient)   
            except Exception as e:
                print('Error in', patient)
                raise e
        return None
        
    def __repr__(self):
        n_patients = len(self)
        
        f_control = lambda x: x.group.lower()=='control'
        n_control = len(self.filter(f_control))
        f_nt1 = lambda x: x.group.lower()=='nt1'
        n_nt1 = len(self.filter(f_nt1))

        return f'SleepSet({n_patients} Patients, {n_control} Control, '\
               f'{n_nt1} NT1)'
        
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
        elif str(type(key)())=='0': # that means it is an int
            return self.patients[key]
        else:
            raise KeyError('Unknown key type:{}, {}'.format(type(key), key))
        return SleepSet(items)
    
    
    def __len__(self):
        """return the number of patients in this set"""
        return len(self.patients)
    
    
    def filter(self, function):
        p_true = []
        for p in self.patients:
            try: 
                is_true = function(p)
                if is_true: p_true.append(p)
            except Exception as e:
                code = p.attrib.get('code', 'unknown')
                print(f'Can\'t filter {code}: {e}')
            
        return SleepSet(p_true)
    
    
    def add(self, patient):
        """
        Inserts a Patient to the SleepSet
        
        :param patient: Either a Patient or a file string to an Unisens object
        """
        if isinstance(patient, Patient):
            self.patients.append(patient)
        else:
            raise ValueError(f'patient must be or Patient, is {type(patient)}')
        return self
    
    def get_feats(self, name):
        feats = [p.get_feat(name) for p in self]
        return np.hstack(feats)
    
    def get_hypnos(self, only_sleeptime=False):
        hypnos = [p.get_hypno(only_sleeptime) for p in self]
        return hypnos
    
    def print(self):
        s = '['
        s += ',\n'.join([str(x) for x in self.patients])
        print(s + ']')
    
    
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
        name = self.attrib.get('code', 'unkown')
        sfreq = int(self.attrib.get('sampling_frequency', -1))
        seconds = int(self.attrib.get('duration', 0))
        length = time.strftime('%H:%M:%S', time.gmtime(seconds))
        gender = self.attrib.get('gender', '')
        age = self.attrib.get('age', -1)
        
        if 'features' in  self:
            nfeats = len(self['feats'])
        else:
            nfeats = 0
        return f'Patient({name}, {length} , {sfreq} Hz, {nfeats} feats, '\
               f'{gender} {age} y)'
    
    def __str__(self):
        return repr(self)
    
    def __init__(self, folder, *args, **kwargs):
        if isinstance(folder, Patient): return None
        if not 'autosave' in kwargs: kwargs['autosave'] = True
        super().__init__(folder, convert_nums=True, *args, **kwargs)

    def __len__(self):
        """
        returns the length of this PSG in seconds
        """
        seconds = int(len(self.data)//(self.sfreq))
        return seconds
        
    def get_hypno(self, only_sleeptime=False, cache=True):
        if cache and hasattr(self, '_hypno'):
            hypno = self._hypno
        else:
            try:
                hypno = self['hypnogram.csv'].get_data()
            except:
                hypno = self['hypnogram_old.csv'].get_data()
            hypno = np.array(list(zip(*hypno))[1])
            self._hypno = hypno
        if only_sleeptime:
            start = np.argmax(np.logical_and(hypno>0 , hypno<5))
            end = len(hypno)-np.argmax(np.logical_and(hypno>0 , hypno<5)[::-1])
            hypno = hypno[start:end]
        return hypno
     
    def get_ecg(self):
        return self['ecg.bin'].get_data().squeeze()

    def get_eeg(self):
        return self['eeg.bin'].get_data().squeeze()
    
    def get_arousals(self):
        """return epochs in which an arousal has taken place"""
        arousals = self['arousals'].get_data()
        
    def get_feat(self, name):
        if isinstance(name, int):
            name = config.feats_mapping[name]
        return self.feats[name].get_data().squeeze()
    

        
    
    """creates a unisens.xml that shows only the features"""
    def write_features_to_unisens(self):
        u = Unisens(folder=self._folder, filename='features.xml')
        
        if 'ecg' in self.feats._parent: 
            ecg = self.feats._parent['ecg'].copy()
            u.add_entry(ecg)
    
        if 'hypnogram' in self.feats._parent: 
            stages = self.feats._parent['hypnogram'].copy()
            u.add_entry(stages)
        
        for entry in self.feats._entries:
            u.add_entry(entry)
        u.save()
    
    def plot(self, channel='eeg', hypnogram=True, axs=None):
        hypnogram = hypnogram * ('hypnogram' in self or 'hypnogram_old.csv' in self)
        plots = 1 + hypnogram
        h_ratio = [0.75,0.25] if hypnogram else [1,] 
        
        if axs is None:
            fig, axs = plt.subplots(plots, 1, 
                                    gridspec_kw={'height_ratios':h_ratio}, 
                                    squeeze=False)
            
        file = ospath.join(self._folder, '/plots/', f'plot_{channel}.png')
        entry =  self[channel]
        signal = entry.get_data()
        if entry.id.endswith('bin'):
            signal = signal[0]
            sfreq = entry.sampleRate
            sleep_utils.specgram_multitaper(signal, int(sfreq), ax=axs[0][0])
            
        elif entry.id.endswith('csv'):
            sfreq = entry.samplingRate
            axs[0][0].plot(signal[0], signal[1])
        if hypnogram: axs[0][0].tick_params(axis='x', which='both', bottom=False,     
                                            top=False, labelbottom=False) 
        plt.title(f'{channel}, {sfreq} Hz')
        
        if hypnogram:
            hypno = self.get_hypno()
            sleep_utils.plot_hypnogram(hypno, ax=axs[-1][0])
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file)
        return file