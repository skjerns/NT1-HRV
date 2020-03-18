# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:16:26 2020

this script creates multitaper spectrograms for
all our record files.

@author: skjerns
"""
from sleep import Patient
import sleep_utils
import ospath
import config as cfg
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':  
    dataset = cfg.folder_edf
    files = ospath.list_files(dataset, exts='edf')
    ax = plt.subplot(1,1,1)
    for file in tqdm(files):
        plt.cla()
        sleep = Patient(file, channel= 'ECG I', verbose=False)
        sleep_utils.specgram_multitaper(sleep.data, sleep.sfreq, ufreq=10, ax=ax)
        plt.title('{} ECG'.format(ospath.basename(file)))
        png_file = file[:-4] + '_ecg.png'
        plt.savefig(png_file)
        
        plt.cla()
        sleep = Patient(file, verbose=False)
        sleep_utils.specgram_multitaper(sleep.data, sleep.sfreq, ufreq=35, ax=ax)
        plt.title('{} EEG'.format(ospath.basename(file)))
        png_file = file[:-4] + '_eeg.png'
        plt.savefig(png_file)