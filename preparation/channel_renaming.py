# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:05:33 2019
@author: Simon

Use this file to rename channel to a common channel name

"""

import sys
sys.path.append("..")
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
           'Fz:A1':     'EEG Fz-A1',
           'EOGl:A1':   'EOGl-A1',
           'EOGl:A2':   'EOG LOC-A2',
           'EOGr:A1':   'EOG ROC-A1',
           'EOGr:A2':   'EOG ROC-A2',
           'EOG1:A2':   'EOG LOC-A2',
           'EOG2:A1':   'EOG ROC-A1',
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
           }




datafolder = 'z:/NT1-HRV'
output = 'z:/renamed'
files = ospath.list_files(datafolder, exts='edf', subfolders=True)

# We rename the channels to have the same names in all recordings
def rename(file):
    header = sleep_utils.read_edf_header(file)
    channels = header['channels']
    new_file = ospath.join(output, ospath.basename(file))
    if ospath.exists(new_file): return

    signal_headers = []
    signals = []
    for ch_nr in tqdm(range(len(channels))):
        signal, signal_header, _ = sleep_utils.read_edf(file, digital=True, 
                                                    ch_nrs=ch_nr, verbose=False)
        ch = signal_header[0]['label']
        ch = ch.replace(':M', ':A')
        if ch in remove: continue
        if ch in ch_mapping.values():
            pass
        elif ch in ch_mapping :
            print('{} to {}'.format(ch, ch_mapping[ch]))
            ch = ch_mapping[ch]
            signal_header[0]['label']=ch
        else:
            print('no mapping for {}'.format(ch))
        signal_headers.append(signal_header[0])
        signals.append(signal.squeeze())

    sleep_utils.write_edf(new_file, signals, signal_headers, header,digital=True)
    
for i, file in enumerate(tqdm(files)):
    rename(file)