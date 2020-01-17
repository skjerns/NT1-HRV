# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:05:33 2019
@author: Simon

Use this file to rename channel to a common channel name

"""
import os
import sys
sys.path.append("..")
import config as cfg
from tqdm import tqdm
import ospath
import sleep_utils

remove = ['C3', 'C4', 'EOGl', 'EOGr', 'M1', 'M2', 'A1', 'A2', 'EOGV', 'Fz', 'Oz',
          'Plethysmogram', 'Pleth', 'Akku', 'Numeric Aux10', 'O1', 'O2', 'FlowPAP', 
          'Druck', 'li Arm', 'Skin potential', 'Numeric Aux9']

ch_mapping = {
           
           'C4:A1':     'EEG C4-A1',
           'C3:A2':     'EEG C3-A2',
           'F4:A1':     'EEG F4-A1',
           'Fz:A2':     'EEGF z-A2',
           'O2:A1':     'EEG O2-A1',
           'Oz:A1':     'EEG Oz-A1',
           'Oz:A2':     'EEG Oz-A2',
           'EEG O1-A2': 'EEG O1-A2',
           'Fz:A1':     'EEG Fz-A1',
           'C4:M1':     'EEG C4-A1',
           'C3:M2':     'EEG C3-A2',
           'F4:M1':     'EEG F4-A1',
           'Fz:M2':     'EEGF z-A2',
           'O2:M1':     'EEG O2-A1',
           'Oz:M1':     'EEG Oz-A1',
           'Oz:M2':     'EEG Oz-A2',
           'Fz:M1':     'EEG Fz-A1',
           'EOGl:A1':   'EOG ROC-A1',
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
           'Licht':'Lux Ambient',
           'Druck Flow':'Druck Flow',
           'Microphone':'Microphone',
           'Akku':'Akku'
           }

# H
nt1_datafolder = 'Z:/NT1-HRV/NT1/ch_renamed'#ospath.join(cfg.data ,'NT1')
files = ospath.list_files(nt1_datafolder, exts='edf')


if __name__ == '__main__':
    for file in tqdm(files):
        new_folder = os.path.join(nt1_datafolder, 'ch_renamed')
        os.makedirs(new_folder, exist_ok=True)
        new_file = os.path.join(new_folder, os.path.basename(file))
        if os.path.isfile(new_file):
            print('Already exists: {}'.format(new_file))
            continue
        sleep_utils.rename_channels(file, new_file=new_file, mapping=ch_mapping)
    print('Files have been created in {}, please move them manually.'.format(new_folder))

