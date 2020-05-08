# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:05:48 2019

This modiule can be used to store machine/user-defined variables.

As this file is hosted on GitHub:
    
DO NOT STORE PRIVACY SENSITIVE INFORMATION IN THIS FILE

All privacy
    
@author: skjerns
"""
import json
import os, sys
from misc import CaseInsensitiveDict
import ospath
import getpass
import platform
from pathlib import Path


# class AttrDict(dict):
#     """
#     A dictionary that allows access via attributes
#     a['entry'] == a.entry
#     """
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self


def get_dropbox_location():
    try:
        json_path = (Path(os.getenv('LOCALAPPDATA'))/'Dropbox'/'info.json').resolve()
    except FileNotFoundError:
        try:
            json_path = (Path(os.getenv('APPDATA'))/'Dropbox'/'info.json').resolve()
        except:
            print('dropbox does not seem to be installed')
            return False

    with open(str(json_path)) as f:
        j = json.load(f)
    
    personal_dbox_path = Path(j['personal']['path'])
    return personal_dbox_path


###############################
#%% USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

dropbox = get_dropbox_location()
if dropbox:
    documents = ospath.join(dropbox, 'nt1-hrv-documents')
    matching = ospath.join(documents, 'matching.csv')
    edfs_invert = ospath.join(documents, 'edfs_invert.csv')
    edfs_discard = ospath.join(documents, 'edfs_discard.csv')
    controls_csv =  ospath.join(documents, 'subjects_control.csv')
    patients_csv =  ospath.join(documents, 'subjects_nt1.csv')

if username == 'nd269' and host=='ess-donatra':
    USER_VAR = 'test123'
    
elif username == 'simon' and host=='desktop-skjerns':
    USER_VAR = 'test456'
    
else:
    print('Username {} on host {} with {} has no configuration.\n'.format(username,host,system) + \
    'please set user specific information in config.py')


    
try: 
    sys.path.append(documents)
    from user_variables import *
except:
    print('It seems like you have not set the documents folder for this '\
          'machine. Please add in config.py. The documents folder contains '\
          'another script user_variables.py where you can set privacy' \
          'sensitive stuff that should not land on github such as dataset '\
          'paths.')

root_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in vars() else ''


############################################
#%% GENERAL LOOKUP TABLES AND VARIABLES
############################################

ecg_channel = 'ECG I'
max_age_diff = 3 # maximum age difference to make a matching

# Conversion for sleep stage names to numbers and back
stage2num = {'WAKE':0, 'WACH':0, 'WK':0,  'N1': 1, 'N2': 2, 'N3': 3, 'N4':3, 'REM': 4,
                 0:0, 1:1, 2:2, 3:3, 4:4, -1:5, 5:5, 'ART': 5, 'A':5, 'ARTEFAKT':5, 'MT':5}
num2stage = {0:'WAKE', 1:'S1', 2:'S2', 3:'SWS', 4:'REM', 5:'Artefact'}


# Mapping from numers to body position for somnoscreen
mapping_body = {1: 'face down', 
                2: 'upright', 
                3: 'left', 
                4: 'right',
                5: 'upside down',
                6: 'face up'}
# reverse as well
mapping_body.update( {v: k for k, v in mapping_body.items()})


# Features mapping to names
# The names should be used to store and access the features in Unisens
# The dictionary is case insensitive
mapping_feats = {1:  'mean_HR',
                 2:  'mean_RR',
                 3:  'detrend_mean_RR',
                 4:  'SDNN',
                 5:  'RR_range',
                 6:  'pNN50',
                 7:  'RMSSD',
                 8:  'SDSD',
                 9:  'RR_log_VLF',
                 10: 'LF',                  # LF power
                 11: 'HF',                  # HF power
                 12: 'LF_HF',               # ratio LF/HF
                 13: 'mean_RR_resp_freq',   # RR mean respiratory frequency
                 14: 'mean_RR_resp_pow',    # RR mean respiratory power
                 15: 'max_phase_HF',        # max phase HF pole
                 16: 'max_mod_HF',          # max mod HF pole
                 # multiscale entr. of RR intervals 1-2 scale 1-10 over 510 sec
                 **dict(zip(range(17,27), [f'multiscale_entropy_1_{s}' for s in range(1,11)])),
                 **dict(zip(range(27,37), [f'multiscale_entropy_2_{s}' for s in range(1,11)])),
                 37: 'RR_DFA',
                 38: 'RR_DFA_short',        # RR DFA short exponent
                 39: 'RR_DFA_long',         # RR DFA long exponent
                 40: 'RR_DFA_all',          # RR DFA all scales
                 41: 'WDFA',                # WDFA over 330 sec
                 42: 'PDFA',                # PDFA non-overlapping segments of 64 heart beats
                 43: 'mean_abs_diff_HR',
                 44: 'mean_abs_diff_RR',
                 45: 'mean_abs_diff_detr_HR', 
                 46: 'mean_abs_diff_detr_RR', 
                 # 47-51: RR percentiles
                 **dict(zip(range(47,52), [f'RR_{i}_perc' for i in (10,25,50,75,90)])),
                 # 52-56: HR percentiles
                 **dict(zip(range(52,57), [f'HR_{i}_perc' for i in (10,25,50,75,90)])),
                 # 47-51: detrended RR percentiles
                 **dict(zip(range(57,62), [f'detr_RR_{i}_perc' for i in (10,25,50,75,90)])),
                 # 52-56: detrended HR percentiles
                 **dict(zip(range(62,67), [f'detr_HR_{i}_perc' for i in (10,25,50,75,90)])),
                 67: 'sample_entropy',      # sample entropy of symbolic binary change in RR interval
                 68: 'ECG_power',           # power of ECG
                 69: 'ECG_4th_power',       # fouth power of ECG
                 70: 'ECG_curve_length',
                 71: 'nonlinear_energy',
                 72: 'hjorth_mobility', 
                 73: 'ECG_complexity',
                 74: 'ECG_peak_pow_psd',
                 75: 'ECG_peak_freq_psd',
                 76: 'ECG_peak_mean_psd',
                 77: 'ECG_peak_media_psd',
                 78: 'spectral_entropy',    # of ECG
                 79: 'hurst_exponent', 
                 80: 'short_phase_coord',   # short phase coordination
                 81: 'long_phase_coord'     # long phase coordination
                 }


mapping_feats.update( {v: k for k, v in mapping_feats.items()}) # reverse as well
mapping_feats = CaseInsensitiveDict(mapping_feats)


mapping_channels = {  # this is the mapping to streamline channel names of different recordings
           'C4:A1':     'EEG C4-A1',
           'C3:A2':     'EEG C3-A2',
           'F4:A1':     'EEG F4-A1',
           'Fz:A2':     'EEG Fz-A2',
           'O2:A1':     'EEG O2-A1',
           'Oz:A1':     'EEG Oz-A1',
           'Oz:A2':     'EEG Oz-A2',
           'EEG O1-A2': 'EEG O1-A2',
           'Fp1:M1':    'EEG Fp1-A1',
           'Fp2:M1':    'EEG Fp2-A1',
           'Fz:A1':     'EEG Fz-A1',
           'C4:M1':     'EEG C4-A1',
           'C3:M2':     'EEG C3-A2',
           'F4:M1':     'EEG F4-A1',
           'Fz:M2':     'EEG Fpz-A2',
           'O1:M2':     'EEG O1:A2',
           'O2:M1':     'EEG O2-A1',
           'Oz:M1':     'EEG Oz-A1',
           'Oz:M2':     'EEG Oz-A2',
           'Fz:M1':     'EEG Fz-A1',
           'EOGl:A1':   'EOG ROC-A1',
           'EOGl:M1':   'EOG LOC-A1',
           'EOGl:A2':   'EOG LOC-A2',
           'EOGl:M2':   'EOG LOC-A2',
           'EOGr:A1':   'EOG ROC-A1',
           'EOGr:M1':   'EOG ROC-A1',
           'EOGr:A2':   'EOG ROC-A2',
           'EOGr:M2':   'EOG ROC-A2',
           'EOG1:A1':   'EOG LOC-A1',
           'EOG1:A2':   'EOG LOC-A2',
           'EOG1:M2':   'EOG LOC-A2',
           'EOG2:A1':   'EOG ROC-A1',
           'EOG2:M1':   'EOG ROC-A1',
           'EOG2:A2':   'EOG ROC-A2',
           'EOG1':      'EOG LOC-A2',
           'EOG2':      'EOG ROC-A1',
           'EOG LOC':   'EOG LOC-A2',
           'EOG ROC':   'EOG ROC-A1',
           'ECG':       'ECG I',
           'EMGchin':      'EMG Chin',
           'EMG EMGTible': 'EMG Chin',
           'EMG EMGTibri': 'EMG Chin',
           'EMG EMGarmle': 'EMG Aux1',
           'EMG EMGarmri': 'EMG Aux2',
           'EMG':  'EMG Chin',
           'EMG1': 'EMG Chin',
           'EMG2': 'EMG Aux1',
           'EMG EMGchin':'EMG Chin',
           'EKG': 'ECG I',
           'PLM l':'PLM l',
           'PLM r':'PLM r',
           'Bein l': 'PLM l',
           'Bein r': 'PLM r',
           'Snore':'Snore',
           'Abdomen':'Abdomen',
           'Thorax':'Thorax',
           'Snoring':'Snore',
           'Schnarc':'Snore',
           'RIP.Thrx':'Thorax',
           'RIP.Abdom':'Abdomen',
           'Abdom':'Abdomen',
           'SPO2':'SpO2',
           'SpO2':'SpO2',
           'Puls':'PulseRate',
           'Pulse':'PulseRate',
           'Airflow':'Flow Patient',
           'Flow':'Flow Patient',
           'Lage':'Body',
           'Body position':'Body',
           'Beweg.':'Accelerometer',
           'Summe Effort':'Effort',
           'Licht':'Lux Ambient',
           'Druck Flow':'Pressure',
           'Druck': 'Pressure',
           'Microphone':'Microphone',
           'Akku':'Akku'
           }