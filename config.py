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
import ospath
import getpass
import platform
from pathlib import Path


class AttrDict(dict):
    """
    A dictionary that allows access via attributes
    a['entry'] == a.entry
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GENERAL CONFIGURATION
###############################

channel_mapping = {  # this is the mapping to streamline channel names of different recordings
           'C4:A1':     'EEG C4-A1',
           'C3:A2':     'EEG C3-A2',
           'F4:A1':     'EEG F4-A1',
           'Fz:A2':     'EEGF z-A2',
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
           'RIP.Thrx':'Effort THO',
           'RIP.Abdom':'Effort ABD',
           'Abdom':'Abdomen',
           'SPO2':'SpO2',
           'SpO2':'SpO2',
           'Puls':'PulseRate',
           'Pulse':'PulseRate',
           'Airflow':'Flow Patient',
           'Flow':'Flow Patient',
           'Lage':'Body',
           'Beweg.':'Accelerometer',
           'Summe Effort':'Effort',
           'Licht':'Lux Ambient',
           'Druck Flow':'Druck Flow',
           'Microphone':'Microphone',
           'Akku':'Akku'
           }

ecg_channel = 'ECG I'
max_age_diff = 3 # maximum age difference to make a matching

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# USER SPECIFIC CONFIGURATION
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
if username == 'nd269' and host=='ess-donatra':
    USER_VAR = 'test123'
    
elif username == 'simon' and host=='desktop-skjerns':
    USER_VAR = 'test456'
    
else:
    print('Username {} on host {} with {} has no configuration.\n'.format(username,host,system) + \
    'please set user specific information in config.py')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# USER SPECIFIC CONFIGURATION
###############################
    
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