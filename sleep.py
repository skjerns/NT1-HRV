# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:53:08 2019

todo: check length of hypno and features

@author: skjerns
"""
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
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter1d

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
    
    def __init__(self, patient_list:list=None, readonly=True):
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
                patient = Patient(patient, readonly=readonly)
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
    
    def summary(self):
        """print a detailed summary of the items in this set"""
        n_all = len(self)
        n_nt1 = len(self.filter(lambda x: x.group=='nt1'))
        n_cnt = len(self.filter(lambda x:  x.group=='control'))
        n_nt1_m = len(self.filter(lambda x: x.group=='nt1' and x.match!=''))
        n_cnt_m = len(self.filter(lambda x:  x.group=='control' and x.match!=''))
        n_ecg = len(self.filter(lambda x: 'ECG' in x))
        n_eeg = len(self.filter(lambda x: 'EEG' in x))
        n_emg = len(self.filter(lambda x: 'EMG' in x))
        n_eog = len(self.filter(lambda x: 'EOG' in x))
        n_lage = len(self.filter(lambda x: 'Body' in x.channels))
        n_thorax = len(self.filter(lambda x: 'thorax' in x))
        
        s = f'{n_all} Patients, {n_nt1} NT1 ({n_nt1_m} matched), {n_cnt} controls ({n_cnt_m} matched)\n'
        s += '\nECG:\t' + ("all" if n_all==n_ecg else f"{n_ecg}/{n_all}")
        s += '\nEEG:\t' + ("all" if n_all==n_eeg else f"{n_eeg}/{n_all}")
        s += '\nEMG:\t' + ("all" if n_all==n_emg else f"{n_emg}/{n_all}")
        s += '\nEOG:\t' + ("all" if n_all==n_eog else f"{n_eog}/{n_all}")
        s += '\nLage:\t' + ("all" if n_all==n_lage else f"{n_lage}/{n_all}")
        s += '\nThorax\t' + ("all" if n_all==n_thorax else f"{n_thorax}/{n_all}")
        print(s)
        
        
        
    def print(self):
        """pretty-print all containing patients in a list"""
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
    def __new__(cls, folder=None, *args, **kwargs):
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
        
        if 'feats' in  self:
            nfeats = len(self['feats'])
        else:
            nfeats = 'None'
        return f'Patient({name}, {length} , {sfreq} Hz, {nfeats} feats, '\
               f'{gender} {age} y)'
    
    def __str__(self):
        return repr(self)
    
    def __init__(self, folder, *args, **kwargs):
        if isinstance(folder, Patient): return None
        if not 'autosave' in kwargs: kwargs['autosave'] = False
        super().__init__(folder, convert_nums=True, *args, **kwargs)

    def __len__(self):
        """
        returns the length of this PSG in seconds
        """
        seconds = int(len(self.data)//(self.sfreq))
        return seconds
         
        
    def get_artefacts(self, only_sleeptime=False,
                      block_window_length=15):
        """
        As some calculations include surrounding epochs, we need to figure
        out which epochs cant be used because their neighbouring epochs
        have an artefact.
        
        block_window_length 0 will only get the annotated artefacts as boolean array
        block_window_length 1 will get the same boolean array but with each neighbour
        seconds block_window_length/2 blocked as well. Kind of like a cellular automata ;-)
        """
        if hasattr(self, '_artefacts_cache'):
            data = self._artefacts_cache
        else:
            try:
                data = list(zip(*self['artefacts'].get_data()))[1]  
            except:
                data = np.zeros(self.epochs_hypno*2)
            self._artefacts_cache = data
            
        data = np.array(data, dtype=bool)
        
        # now repeat to get on a per-second-basis
        data = np.repeat(data, 15)
        
        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            data = data[self.sleep_onset:self.sleep_offset]
            
        block_window_length -= 15 # substract window middle
        if block_window_length>0:
            data = binary_dilation(data, structure=[True,True,True], 
                                   iterations=block_window_length)
        
        # we are very strict. If there is a single second of artefact, 
        # we discard the whole epoch.
        data = data.reshape([-1,30]).max(1)
        return data

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
            
        self.sleep_onset = np.argmax(np.logical_and(hypno>0 , hypno<5))*30
        self.sleep_offset = len(hypno)*30-np.argmax(np.logical_and(hypno>0 , hypno<5)[::-1])*30
        if only_sleeptime:
            hypno = hypno[self.sleep_onset//30:self.sleep_offset//30]
        return hypno
    
     
    def get_ecg(self, only_sleeptime=False):
        data = self['ecg.bin'].get_data().squeeze()
        
        if only_sleeptime:
            sfreq = int(self.ecg.sampleRate)
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            data = data[self.sleep_onset*sfreq:self.sleep_offset*sfreq]
        return data

    def get_eeg(self, only_sleeptime=False):
        data = self['eeg.bin'].get_data().squeeze()
        if only_sleeptime:
            sfreq = int(self.eeg.sampleRate)
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            data = data[self.sleep_onset*sfreq:self.sleep_offset*sfreq]
        return data
    
    def get_arousals(self, only_sleeptime=False):
        """return epochs in which an arousal has taken place"""
        try:
            epochs = self['arousals.csv'].get_data()
        except:
            log.warn(f'{self.code} has no arousal file')
            return(np.array([0]))
        arousals = set()
        for epoch, length in epochs:
            arousals.add(epoch)
            i = 1
            while length>30:
                arousals.add(epoch+i)
                i+=1
                
        data = np.array(sorted(list(arousals)))       
        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            data = data-self.sleep_onset//30
        return data
    
    
    def get_feat(self, name, only_sleeptime=False, cache=True):
        if isinstance(name, int):
            name = 'feats/' + config.feats_mapping[name] + '.csv'
            
        if cache and hasattr(self, f'{name}'):
            feat = self.__dict__[f'{name}']
        else:      
            
            feat = np.array(self.feats[name].get_data())[:,1]
            self.__dict__[f'{name}'] = feat
            
        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            feat = feat[self.sleep_onset//30:self.sleep_offset//30]
        return feat
    

        
    
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
        

    
    def plot(self, channels='eeg', hypnogram=True, axs=None):
        with plt.style.context('default'):
            hypnogram = hypnogram * ('hypnogram' in self or 'hypnogram_old.csv' in self)
            if isinstance(channels, str): channels = [channels]
            n_chs = len(channels)
            plots = n_chs + hypnogram
            
            h_ratio = [*[0.75/n_chs]*n_chs,0.25] if hypnogram else [((0.75/n_chs)*n_chs)]
            
            if axs is None:
                fig, axs = plt.subplots(plots, 1, 
                                        gridspec_kw={'height_ratios':h_ratio}, 
                                        squeeze=False)
            axs = axs.flatten()
            file = ospath.join(self._folder, '/plots/', f'plot_{"_".join(channels)}.png')
            for i, channel in enumerate(channels):
                if channel in self:
                    entry =  self[channel]
                    signal = entry.get_data()
                    if entry.id.endswith('bin'):
                        signal = signal[0]
                        sfreq = entry.sampleRate
                        sleep_utils.specgram_multitaper(signal, int(sfreq), ax=axs[i])
                        
                    elif entry.id.endswith('csv'):
                        sfreq = entry.samplingRate
                        signal = list(zip(*signal))
                        axs[i].plot(signal[0], signal[1])
                else:
                    signal = self.get_feat(channel)
                    smoothed = gaussian_filter1d(signal, sigma=3.5)
                    sfreq = '1/30'
                    axs[i].plot(np.arange(len(signal)), signal, linewidth=0.7, alpha=0.6)
                    axs[i].plot(np.arange(len(signal)), smoothed, c='b')
                    axs[i].set_xlim([0,len(signal)])
                axs[i].set_title(channel)
                
            for ax in axs[:-1]:
                ax.tick_params(axis='x', which='both', bottom=False,     
                                                top=False, labelbottom=False) 
            
            formatter = FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
            axs[-1].xaxis.set_major_formatter(formatter)
            
            if hypnogram:
                artefacts = self.get_artefacts()
                hypno = self.get_hypno()
                sleep_utils.plot_hypnogram(hypno, ax=axs[-1])
                for i, is_art in enumerate(artefacts):
                    plt.plot([i*30,(i+1)*30],[0.2, 0.2],c='red', 
                             alpha=0.75*is_art, linewidth=1)
            plt.suptitle(f'Plotted: {channels}, {sfreq} Hz', y=1)
            plt.pause(0.01)
            plt.tight_layout()
            plt.pause(0.01)
            plt.savefig(file)
            os.makedirs(os.path.dirname(file), exist_ok=True)

        return file