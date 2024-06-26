# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:53:08 2019

todo: check length of hypno and features

@author: skjerns
"""
import config
import logging as log
from tqdm import tqdm
import numpy as np
import ospath
import os
import re
import time
import shutil
import sleep_utils
import features
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from boltons.funcutils import wraps
from unisens import Unisens, CustomEntry
from unisens.utils import make_key
from joblib import Parallel, delayed

log.basicConfig()
log.getLogger().setLevel(log.INFO)

def natsort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def error_handle(func):
    """
    a small wrapper that will print the Patient folder
    in case an error appears in the wrapped class method
    """
    @wraps(func)
    def print_error_wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'### ERROR in Patient {self}')
            raise e
    print_error_wrapper.__doc__ = func.__doc__
    return print_error_wrapper



class SleepSet():
    """
    A SleepSet is a container for several Patients, where each Patient
    corresponds to one PSG recording (ie. an EDF file). This container
    makes it easier to do bulk operations, feature extraction etc.
    on a whole set of Patients.
    """

    def __init__(self, patient_list:list=None, readonly=False):
        """
        Load a list of Patients (edf format). Resample if necessary.

        :param patient_list: A list of strings pointing to Patients
        """
        if isinstance(patient_list, str):
            patient_list = ospath.list_folders(patient_list)
        assert isinstance(patient_list, list), 'patient_list must be type list'
        self.patients = []

        # return if list is empty
        if len(patient_list)==0: return None

        # must be either Patients or strings to make patients from.
        all_patients = all(['Patient' in str(type(x)) for x in patient_list])
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
        groups = set([p.get_attrib('group', 'None') for p in self])
        counts = []
        for group in groups:
            fn = lambda p: p.get_attrib('group')==group
            count = f'{len(self.filter(fn))} {group}'
            counts.append(count)
        return f'SleepSet({n_patients} Patients, ' + ', '.join(counts) + ')'

    def __iter__(self):
        """
        iterate through all patients in this set
        """
        return self.patients.__iter__()

    def __contains__(self, key):
        if 'Patient' in str(type(key)):
            key = key.code
        try:
            self[key]
            return True
        except:
            return False


    def __getitem__(self, key):
        """
        grant access to the set with slices and indices and keys,
        as well as with patient codes
        """
        # if it's one item: return this one item
        if type(key)==slice:
            items = self.patients.__getitem__(key)
        elif type(key)==np.ndarray:
            items = [self.patients.__getitem__(i) for i in key]
        elif str(type(key)())=='0': # that means it is an int
            return self.patients[key]
        # this means we want to access via code, eg '123_456'
        elif isinstance(key, str) and key.count('_')==1:
            for p in self:
                if hasattr(p, 'code') and p.code==key:
                    return p
            raise KeyError(f'Patient {key} not found')
        else:
            raise KeyError('Unknown key type:{}, {}'.format(type(key), key))
        return SleepSet(items)

    def __len__(self):
        """return the number of patients in this set"""
        return len(self.patients)

    def filter(self, function, verbose=True):
        p_true = []
        for p in self.patients:
            try:
                is_true = function(p)
                if is_true: p_true.append(p)
            except Exception as e:
                code = p.attrib.get('code', 'unknown')
                if verbose:
                    print(f'Can\'t filter {code}: {e}')

        return SleepSet(p_true)


    def compute_features(self, names=None, wsize=None, step=None, offset=None,
                         overwrite=False, n_jobs=-1):
        Parallel(n_jobs=n_jobs)(delayed(Patient.compute_features)\
                                (p, names=names, wsize=wsize, step=step,
                                 offset=offset, overwrite=overwrite) for p in tqdm(self))


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

    def stratify(self, filter=lambda x:True):
        """
        Return the subset where each patient has a match.
        After stratification, there should be an equal number of
        nt1 and controls in the subset and each patient should have exactly
        one match
        An additional filter can be applied on the fly.

        :param filter: an additional filter lambda to be applied
        returns two SleepSets: NT1 and Matches
        """
        subset = self.filter(filter)
        new_set = SleepSet([])
        for p in subset:
            if p.__dict__.get('match', 'notfound') in subset:
                new_set.add(p)
        assert len(new_set.filter(lambda x: x.group=='nt1')) == len(new_set.filter(lambda x: x.group=='control'))
        return new_set


    def get_feats(self, name, only_sleeptime=False, wsize=300, step=30,
                   offset=None, cache=True, only_clean=True,
                   sleep_onset_offset=None):
        feats = [p.get_feat(name,
                            only_sleeptime=only_sleeptime,
                            wsize=wsize,
                            step=step,
                            offset=offset,
                            cache=cache,
                            only_clean=only_clean,
                            sleep_onset_offset=sleep_onset_offset) for p in self]
        return feats

    def get_hypnos(self, only_sleeptime=False):
        hypnos = [p.get_hypno(only_sleeptime) for p in self]
        return hypnos


    def summary(self, verbose=True):
        """print a detailed summary of the items in this set"""
        from prettytable import PrettyTable


        # a meta function to filter both groups in one call
        group_names = set([p.group for p in self])
        filtg = lambda func: [len(self.filter(lambda x: func(x) and x.group==group, verbose=False)) for group in group_names]

        n_all = len(self)
        n_both = filtg(lambda x: True)
        n_matched = filtg(lambda x: x.match!='')
        n_male = filtg(lambda x: x.gender=='male')
        n_female = filtg(lambda x: x.gender=='female')
        n_no_gender = filtg(lambda x: not hasattr(x, 'gender'))

        n_ecg = filtg(lambda x: 'ECG' in x)
        n_eeg = filtg(lambda x: 'EEG' in x)
        n_feats = filtg(lambda x: 'feats' in x and len(x.feats)>0)
        n_hypno = filtg(lambda x: len(x.get_hypno())>0)

        # get sampling frequency of ecg
        ecg_hz = []
        for group in group_names:
            sfreqs = [p.ecg.sampleRate for p in self.filter(lambda x: x.group==group, verbose=False)]
            ecg_counts = list(zip(*[list(y) for y in np.unique(sfreqs, return_counts=True)]))
            n_ecg_hz = '\n'.join([f'{hz} Hz (n={n})' for hz,n in ecg_counts])
            ecg_hz.append(n_ecg_hz)

        n_dataset_mnc = filtg(lambda x: x.get_attrib('dataset')=='mnc')
        n_dataset_other =  filtg(lambda x: x.get_attrib('dataset')!='mnc')

        cohorts = set([p.get_attrib('cohort', 'none') for p in self])

        t = PrettyTable(['Name', *group_names, 'Comment'])
        t.align['Name'] = 'l'
        t.align['Comment'] = 'l'
        t.add_row(['Group', *n_both, f'total: {n_all}'])
        t.add_row(['Male', *n_male, f'total: {sum(n_male)}'])
        t.add_row(['Female', *n_female, f'total: {sum(n_female)}'])
        t.add_row(['No gender', *n_no_gender, f'total: {sum(n_no_gender)}'])
        t.add_row(['Matched', *n_matched, f'total: {sum(n_matched)}'])
        t.add_row(['Has ECG',*n_ecg, f'missing: {n_all-sum(n_ecg)}'])
        t.add_row(['Has EEG', *n_eeg, f'missing: {n_all-sum(n_eeg)}'])
        t.add_row(['Has feats', *n_feats, f'missing: {n_all-sum(n_feats)}'])
        t.add_row(['Has hypno', *n_hypno, f'missing: {n_all-sum(n_hypno)}'])
        t.add_row(['ECG Hz', *ecg_hz, ''])
        t.add_row(['Dataset', *['' for _ in group_names], ''])
        t.add_row(['---mnc', *n_dataset_mnc, f'total: {sum(n_dataset_mnc)}'])
        t.add_row(['---other', *n_dataset_other, f'total: {sum(n_dataset_other)}'])
        t.add_row(['', *['' for _ in group_names], ''])
        t.add_row(['Cohort  ', *['' for _ in group_names], ''])
        for cohort in cohorts:
            n_cohort = filtg(lambda x: x.get_attrib('cohort', 'other')==cohort)
            t.add_row([f'---{cohort}', *n_cohort, f'total: {sum(n_cohort)}'])

        if verbose:
            print(t)
        return t

    def print(self):
        """pretty-print all containing patients in a list"""
        s = '['
        s += ',\n'.join([str(x) for x in self.patients])
        print(s + ']')


class Patient(Unisens):
    """
    attributes
        self.sleep_onset  = onset of first non-W and non-5 in seconds
        self.sleep_offset = last sleep stage in seconds

    A Patient contains the data of one unisens data structure.
    It facilitates the automated extraction of features, loading of data
    and hypnogram as well as visualization of the record, plus statistical
    analysis. Many of its subfunctions are inherited from Unisens
    """

    def __new__(cls, folder=None, *args, **kwargs):
        """
        If this patient is initialized with a Patient, just return this Patient
        """
        if 'Patient' in str(type(folder)): return folder
        return object.__new__(cls)

    def __repr__(self):
        name = self.attrib.get('code', 'unkown')
        sfreq = int(self.attrib.get('sampling_frequency', -1))
        seconds = int(self.attrib.get('duration', 0))
        length = time.strftime('%H:%M', time.gmtime(seconds))
        gender = self.attrib.get('gender', 'nogender')
        group = self.attrib.get('group','nogroup')
        age = self.attrib.get('age', -1)

        if 'feats' in  self:
            nfeats = len(self['feats'])
        else:
            nfeats = 'None'
        return f'Patient({name} ({group}), {length}, {sfreq} Hz, {nfeats} feats, '\
               f'{gender} {age}y)'

    def __str__(self):
        return repr(self)

    def __init__(self, folder, *args, **kwargs):
        if isinstance(folder, Patient): return None
        if not 'autosave' in kwargs: kwargs['autosave'] = False
        super().__init__(folder, convert_nums=True, *args, **kwargs)

    @property
    def sleep_onset(self):
        if not hasattr(self, '_sleep_onset'):
            self.get_hypno()
        return self._sleep_onset

    @sleep_onset.setter
    def sleep_onset(self, value):
        self._sleep_onset = value

    @property
    def sleep_offset(self):
        if not hasattr(self, '_sleep_offset'):
            self.get_hypno()
        return self._sleep_offset

    @sleep_offset.setter
    def sleep_offset(self, value):
        self._sleep_offset = value

    @error_handle
    def compute_features(self, names=None, wsize=None, step=None, offset=None,
                         overwrite=True):
        """
        A helper function that computes all features.
        Useful to overwrite or recompute certain features
        but also to just create all features.
        """
        _readonly = self._readonly
        self._readonly = False
        if wsize is None:
            wsize = config.default_wsize
        if step is None:
            step = config.default_step
        if offset is None:
            offset = self.get_attrib('use_offset', config.default_offset)
        if names is None:
            names = list(zip(*config.mapping_feats.items()))[0]
        for name in names:
            # if there is no function for this feature name
            # we skip the calculation of this feature
            # it might not be implemented yet.
            if not name in features.__dict__: continue
            feat_name = f'feats/{name}-{int(wsize)}-{int(step)}-{int(offset)}.npy'
            if overwrite and feat_name in self.feats:
                log.debug('Remove {feat_name}')
                self.feats.remove_entry(feat_name)
            log.debug('create {feat_name}')
            try: self.get_feat(name, wsize=wsize, step=step, offset=offset, cache=False)
            except Exception as e: print(e, repr(e))
        self._readonly = _readonly
        return None


    def get_RRi(self, only_sleeptime=False, offset=None, sleep_onset_offset=None,
                cache=True):
        if offset is None:
            offset = bool(self.get_attrib('use_offset', config.default_offset))
        if sleep_onset_offset is None: sleep_onset_offset = 0
        assert isinstance(offset, bool), 'offset must be boolean not int/float'
        if cache and hasattr(self, '_cache_RRi'):
            RRi = self._cache_RRi
        else:
            RRi = self.feats.get_data()['Data']['RRi']

        if cache:
            self._cache_RRi = RRi

        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            sleep_onset = self.sleep_onset*4 + sleep_onset_offset*4
            slee_offset = self.sleep_offset*4
            RRi = RRi[sleep_onset:slee_offset]
        return RRi


    @error_handle
    def get_RR(self, only_sleeptime=False, offset=None, sleep_onset_offset=None,
               cache=True):
        """
        Retrieve the RR peaks and the T_RR, which is their respective positions

        :param offset: whether to padd the RR to 30 second epochs

        returns: (T_RR, RR)
        """
        if sleep_onset_offset is None: sleep_onset_offset = 0
        if offset is None:
            offset = self.get_attrib('use_offset', config.default_offset)
        offset = bool(offset)
        assert isinstance(offset, bool), f'offset must be boolean not {type(offset)}'

        if cache and \
            ((cached:=self.feats.__dict__.get('_cache_RR')) is not None):
            log.debug('Loading cached RR')
            T_RR, RR = cached
            # already loaded within this session

        elif 'feats/RR.npy' in self.feats and 'feats/T_RR.npy' in self.feats:
            log.debug('Loading saved RR')
            # previously loaded
            RR = self.feats.RR.get_data()
            T_RR = self.feats.T_RR.get_data()
            self.feats._cache_RR = (T_RR.copy(), RR.copy())

        else:
            log.debug('extracting RR from pkl-file')
            # never loaded, need to extract
            data = self['feats.pkl'].get_data()['Data']
            T_RR = data['T_RR'] - self.startsec
            RR = data['RR']
            self.feats._cache_RR = (T_RR.copy(), RR.copy())
            _readonly = self._readonly
            self._readonly = False
            CustomEntry('feats/RR.npy', parent=self.feats).set_data(RR)
            CustomEntry('feats/T_RR.npy', parent=self.feats).set_data(T_RR)
            self.save()
            self._readonly = _readonly

        # the hypnograms from Domino always start at :00 or :30
        # e.g. if the recording starts at 05:05:25, the first hypno epoch starts
        # at 05:05:00, that means 25 seconds are added in the beginning.
        # therefore we need to add this offset to the beginning.
        # additionally domino does not count the last epoch if its not full.
        # generally there seems to be no coherent rule on how Domino
        # handles the end of the recording. Therefore we tell our algorithm
        # how many epochs we have in the hypnogram and it will truncate
        # the features accordingly.
        if offset:
            if self.startsec==0:
                log.error(f'startsec is 0, are you sure this is correct? {self._folder}')
            # this is how many seconds we need to add
            start_at = self.startsec//30*30 - self.startsec
            if start_at>0:
                log.warning(f'positive RR padding for {self._folder}')
                idx = np.argmax(T_RR>start_at)
                T_RR = T_RR - int(T_RR[idx])
                T_RR = T_RR[idx:]
                RR = RR[idx:]
            elif start_at<0:
                log.debug(f'Shifting to start at {start_at} seconds to fill first epoch')
                T_RR_pad = list(range(abs(start_at)))
                RR_pad = [1] * abs(start_at) # dummy RR peaks
                T_RR += abs(start_at)
                T_RR = np.hstack([T_RR_pad, T_RR])
                RR = np.hstack([RR_pad, RR])
            else:
                log.debug('No padding to fill epoch')

        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            start = np.argmax(T_RR >= self.sleep_onset + sleep_onset_offset)
            stop = np.argmax(T_RR >= self.sleep_offset + sleep_onset_offset)
            if stop==0:
                stop = len(T_RR)

            T_RR = T_RR[start:stop]
            RR = RR[start:stop-1]
        assert len(T_RR)==len(RR)+1, 'T_RR does not fit to RR, seems wrong. {len(T_RR)}!={len(RR)+1}'


        return np.array(T_RR), np.array(RR)


    @error_handle
    # @profile
    def get_artefacts(self, only_sleeptime=False, wsize=300, step=30,
                      offset=None, sleep_onset_offset=None, cache=True):
        """
        As some calculations include surrounding epochs, we need to figure
        out which epochs cant be used because their neighbouring epochs
        have an artefact.
        """
        if sleep_onset_offset is None: sleep_onset_offset = 0
        if offset is None:
            offset = bool(self.get_attrib('use_offset', config.default_offset))
        assert wsize in [30, 300], 'Currently only 30 and 300 are allowed as artefact window sizes, we didnt define other cases yet.'
        if step is None: step = wsize

        # this is th artefact name including the parameters
        # this way we can store several versions of the artefacts
        # calculated for different parameters.
        art_name = f'artefacts-{int(wsize)}-{int(step)}-{int(offset)}.npy'
        cache_name = f'_cache_{art_name}'

        ### now some caching tricks to speed up loading of features
        # if cached, reload this cached version bv
        if cache and \
            ((art:=self.feats.__dict__.get(cache_name, None)) is not None):
            log.debug('Loading cached artefacts')
            # already loaded during assignment expression
        # if not cached, but already computed, load computed version
        elif art_name in self:
            log.debug('Loading previously saved artefacts')
            art = self[art_name].get_data()
            # save for caching purposes
            if cache:
                self.feats.__dict__[cache_name] = art
        # else: not computed and not cached, compute this feature now.
        else:
            # receive RRs to calculate artefacts on this
            T_RR, RR = self.get_RR(offset=offset, cache=cache)
            # calculate artefacts given these RRs.
            log.debug('Calculating artefacts')
            hypno = self.get_hypno(cache=cache)
            art = np.array(features.artefact_detection(T_RR,RR, wsize, step, expected_nwin=len(hypno)))

            # we need to change the readability of this Patient
            # to store newly created features.
            _readonly = self._readonly
            self._readonly = False
            entry = CustomEntry(art_name, parent=self).set_data(art)
            # also save the parameters just in case
            entry.wsize = wsize
            entry.step = step
            entry.offset = offset
            self.save()
            self._readonly = _readonly

            # save for caching purposes
            if cache:
                self.feats.__dict__[cache_name] = art

        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            sleep_onset = self.sleep_onset//step + sleep_onset_offset//step
            sleep_offset =  self.sleep_offset//step
            art = art[sleep_onset:sleep_offset]
        return art

    @error_handle
    # @profile
    def get_hypno(self, only_sleeptime=False, sleep_onset_offset=None,
                  cache=True):
        if sleep_onset_offset is None: sleep_onset_offset = 0

        if cache and hasattr(self, '_cache_hypno'):
            hypno = self._cache_hypno
        else:
            try:
                hypno = self['hypnogram.csv'].get_data()
            except:
                hypno = self['hypnogram_old.csv'].get_data()
            hypno = np.array(list(zip(*hypno))[1])
            self._cache_hypno = hypno

        # safety precaution: set first and last epoch to ARTEFACT
        hypno[0] = 5
        hypno[-1] = 5

        # calculate the sleep onset and sleep offset seconds.
        self.sleep_onset = np.argmax(np.logical_and(hypno>0 , hypno<5))*30
        self.sleep_offset = len(hypno)*30-np.argmax(np.logical_and(hypno>0 , hypno<5)[::-1])*30
        if only_sleeptime:
            sleep_onset = self.sleep_onset//30 + sleep_onset_offset//30
            sleep_offset =  self.sleep_offset//30
            hypno = hypno[sleep_onset:sleep_offset]
        return hypno

    @error_handle
    def get_ecg(self, only_sleeptime=False, offset=None,
                sleep_onset_offset=None):
        if sleep_onset_offset is None: sleep_onset_offset = 0

        if offset is None:
            offset = bool(self.get_attrib('use_offset', config.default_offset))
        data = self.get_signal('ECG', offset=offset)

        if only_sleeptime:
            sfreq = int(self.ecg.sampleRate)
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            sleep_onset = self.sleep_onset//sfreq + sleep_onset_offset//sfreq
            sleep_offset =  self.sleep_offset//sfreq
            data = data[sleep_onset:sleep_offset]
        return data


    @error_handle
    def get_signal(self, name='eeg', stage=None, only_sleeptime=False,
                   offset=None, sleep_onset_offset=None):
        """
        get values of a SignalEntry

        :name name of the SignalEntry
        :param stage: only return values for this sleep stage
        :param only_sleeptime: only get values after first sleep until last sleep epoch
        :param offset: only return from full epoch (see drive->preprocessing->offset of hypnogram)
        """
        if sleep_onset_offset is None: sleep_onset_offset = 0

        if offset is None:
            offset = bool(self.get_attrib('use_offset', config.default_offset))
        if sleep_onset_offset is None:
            sleep_onset_offset = 0
        data = self[f'{name}.bin'].get_data().squeeze()
        sfreq = int(self[name].sampleRate)

        if offset and hasattr(self, 'startsec'):
            if self.startsec==0:
                log.error(f'startsec is 0, are you sure this is correct? {self._folder}')
            # this is how much we pad in the beginning to fill the first
            # epoch, as hypnograms start at :00 or :30, but signals start at e.g.
            # :15, we add 15 seconds of signal to round up to 00/30
            start_at = self.startsec - self.startsec//30*30
            if start_at<0: log.error('negative padding should not be possible')
            data = np.pad(data, pad_width=[start_at*sfreq, 0], mode='symmetric')

        n_epochs = len(self.get_hypno())
        data = data[:sfreq*n_epochs*30]

        if only_sleeptime:
            sleep_onset = self.sleep_onset//sfreq + sleep_onset_offset//sfreq
            sleep_offset =  self.sleep_offset//sfreq
            data = data[sleep_onset:sleep_offset]

        if stage is not None:
            hypno = self.get_hypno(only_sleeptime=only_sleeptime)
            mask = np.repeat(hypno, sfreq*30)==stage
            data = data[mask[:len(data)]]
        return data


    @error_handle
    def get_arousals(self, only_sleeptime=False):
        """return epochs in which an arousal has taken place"""
        try:
            epochs = self['arousals.csv'].get_data()
            if len(epochs)==0:
                return(np.array([0]))
        except:
            log.warn(f'{self.code} has no arousal file')
            return(np.array([0]))
        arousals = np.array(list(zip(*epochs))[0])

        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno()
            arousals = arousals[np.argmax(arousals>self.sleep_onset//30):]
        return arousals

    @error_handle
    # @profile
    def get_feat(self, name, only_sleeptime=False, wsize=300, step=30,
                 offset=None, cache=True, only_clean=True,
                 sleep_onset_offset=None):
        """
        Returns the given feature with the chosen parameters.

        ## Features:
        On-the-fly-calculation:
            If the feature is not yet computed we copmute it.
        Caching:
            Features will be kept in memory to reload them quickly.
        """
        if sleep_onset_offset is None: sleep_onset_offset = 0

        if offset is None: offset = self.get_attrib('use_offset', 1)
        if step is None: step = wsize
        if isinstance(name, int):
            name =  config.mapping_feats[name]
        assert name in features.__dict__, \
            f'{name} is not present in features.py. Please check feature name'

        # this is th feature name including the parameters
        # this way we can store several versions of the features
        # calculated for different parameters.
        feat_name = f'feats/{name}-{int(wsize)}-{int(step)}-{int(offset)}.npy'
        feat_id = make_key(feat_name)
        cache_name = f'_cache_{feat_name}'

        # now some caching tricks to speed up loading of features
        # if cached, reload this cached version
        if cache and cache_name in self.feats.__dict__:
            log.debug(f'Loading cached {feat_name}')
            feat = self.feats.__dict__[cache_name]

        # if not cached, but already computed, load computed version
        elif self.feats.__dict__.get(feat_id) is not None and \
            os.path.isfile(self.feats.__dict__.get(feat_id)._filename):
            log.debug(f'Loading saved {feat_name}.npy')
            feat = self.feats.__dict__[feat_id].get_data()

        # else: not computed and not cached, compute this feature now.
        else:
            # receive RRs to calculate feature on
            T_RR, RR = self.get_RR(offset=offset, cache=cache)
            hypno = self.get_hypno(cache=cache)
            RR_windows = features.extract_RR_windows(T_RR, RR, wsize=wsize,
                                                     step=step, pad=True,
                                                     expected_nwin=len(hypno))
            log.debug(f'Calculating {feat_name}')
            # retrieve the function handle from functions.py
            # there should be a function with this name present there.
            feat_func = features.__dict__[name]
            try:
                # the patient argument is ignored by most functions
                feat = feat_func(RR_windows, p=self)
                feat = np.array(feat)

                if len(feat)<5:
                    raise Exception(f'Created feature is too small: {len(feat)}')
                # we need to change the readability of this Patient
                # to store newly created features.
                _readonly = self._readonly
                self._readonly = False
                entry = CustomEntry(feat_name, parent=self.feats).set_data(feat)
                # also save the parameters just in case
                entry.wsize = wsize
                entry.step = step
                entry.offset = offset
                self.save()
                self._readonly = _readonly
            except Exception as e:
                import traceback
                # if there was an error we do not save the values.
                log.error(f'{self}, Cant create feature {feat_name}: ' + repr(e))
                traceback.print_exc()

                feat = np.empty(len(RR_windows))
                feat.fill(np.nan)

        # save for caching purposes
        self.feats.__dict__[f'_cache_{feat_name}'] = feat

        if only_clean:
            art = self.get_artefacts(only_sleeptime=False, wsize=wsize, step=step,
                                     offset=offset)
            feat[art] = np.nan

        if only_sleeptime:
            if not hasattr(self, 'sleep_onset'): self.get_hypno(cache=cache)
            sleep_onset = self.sleep_onset//step + sleep_onset_offset//step
            sleep_offset =  self.sleep_offset//step
            feat = feat[sleep_onset:sleep_offset]
        return feat


    @error_handle
    def reset(self):
        """
        Will delete all on-the-fly computed variables
        and reset all caches.
        removed will be all extracted features, RR intervals, artefacts
        """
        assert not self._readonly, 'Can\'t reset readonly Patient'
        for var in list(self.feats.__dict__):
            if var.startswith('_cache'):
                del self.feats.__dict__[var]
        for feat in list(self.feats):
            self.feats.remove_entry(feat.id)
            if os.path.exists(feat._filename):
                os.remove(feat._filename)
        for entry in list(self):
            if entry.id.startswith('artefacts-'):
                self.remove_entry(entry.id)
                if os.path.exists(entry._filename):
                    os.remove(entry._filename)
        try:
            if os.path.isdir(self._folder + '/feats/'):
                shutil.rmtree(self._folder + '/feats/')
        except Exception as e:
            print(e)
        self.save()
        return self

    @error_handle
    def write_features_to_unisens(self):
        """creates a unisens.xml that shows only the features"""
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

    @error_handle
    def plot(self, markers={}, **kwargs):
        from viewer.viewer import ECGPlotter
        data = self.ecg.get_data().squeeze()
        sfreq = self.ecg.sampleRate
        offset = self.get_attrib('use_offset', 0)
        RRs = self.get_RR(offset=offset, cache=False)[0]
        markers.update({'RRs': RRs})
        default_kwargs = {'nrows':2, 'ncols':1, 'interval':60}
        default_kwargs.update(kwargs)
        ECGPlotter(data, sfreq, markers=markers, verbose=False, **default_kwargs)

    @error_handle
    def spectogram(self, channels='eeg', hypnogram=True, fig=None, saveas=None,
                   **kwargs):
        with plt.style.context('default'):
            hypnogram = hypnogram * ('hypnogram' in self or 'hypnogram_old.csv' in self)
            if isinstance(channels, str): channels = [channels]
            n_chs = len(channels)
            plots = n_chs + hypnogram

            h_ratio = [*[0.75/n_chs]*n_chs,0.25] if hypnogram else [((0.75/n_chs)*n_chs)]

            if fig is None:
                fig = plt.figure()
            axs = fig.subplots(plots, 1, gridspec_kw={'height_ratios':h_ratio}, squeeze=False)
            axs = axs.flatten()

            for i, channel in enumerate(channels):
                ax = axs[i]
                if channel in self:
                    entry =  self[channel]
                    signal = entry.get_data().squeeze()
                    if signal.ndim>1: signal = signal[0]
                    sfreq = int(entry.sampleRate)
                    if sfreq < 10:
                        spec, freqs, _, _ = ax.specgram(signal, Fs=sfreq)
                    else:
                        sleep_utils.specgram_multitaper(signal, sfreq=sfreq, ax=ax, **kwargs)
                else:
                    raise ValueError(f'Entry {channel} not found')

                ax.set_title(channel)

            for ax in axs[:-1]:
                ax.tick_params(axis='x', which='both', bottom=False,
                                                top=False, labelbottom=False)

            formatter = FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
            axs[-1].xaxis.set_major_formatter(formatter)

            if hypnogram:
                offset = self.get_attrib('use_offset', 1)
                artefacts = self.get_artefacts(offset=offset)
                hypno = self.get_hypno()
                labeldict = {0: 'Wake', 4:'REM', 1:'S1', 2:'S2', 3:'SWS', 5:'Artefact'}
                sleep_utils.plot_hypnogram(hypno, ax=axs[-1], labeldict=labeldict)
                for i, is_art in enumerate(artefacts):
                    plt.plot([i*30,(i+1)*30],[0.2, 0.2],c='red',
                             alpha=0.75*is_art, linewidth=1)
            plt.suptitle(f'Plotted: {channels}, {sfreq} Hz', y=1)
            plt.pause(0.01)
            plt.tight_layout()
            plt.pause(0.01)
            file = ospath.join(self._folder, '/plots/', f'plot_{"_".join(channels)}.png')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            if saveas is not False: plt.savefig(file)
            if saveas:
                os.makedirs(os.path.dirname(saveas), exist_ok=True)
                plt.savefig(saveas)

        return fig, axs
