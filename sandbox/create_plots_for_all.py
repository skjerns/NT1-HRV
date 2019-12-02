# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:29:13 2019

@author: SimonKern
"""
import numpy as np
from tqdm import tqdm
from plotting import specgram_multitaper
import ospath
from SleepData import SleepRecord
import matplotlib.pyplot as plt


# alle histogramme plotten um zu sehen wie die noise mässig verteilt sind
datafolder = '../../'
files = ospath.list_files(datafolder, subfolders=True, exts='edf')
for file in tqdm(files):
    sr = SleepRecord(file, channel='ECG I')
    data = sr.raw
    mesh = specgram_multitaper(data, sr.sfreq, show_plot=False)
    plt.hist(mesh.ravel(), bins=np.arange(0,1,0.01), alpha=0.1)
    plt.show()
    plt.pause(0.1)
    
plt.savefig('distribution_all.png')
# plot has been saved at distribution_all.png
#%%
plt.close('all')
datafolder = '../../NT1'
files = ospath.list_files(datafolder, subfolders=True, exts='edf')
for file in tqdm(files):
    sr = SleepRecord(file)
    data = sr.raw
    mesh = specgram_multitaper(data, sr.sfreq, show_plot=False)
    plt.hist(mesh.ravel(), bins=np.arange(0,1,0.01), alpha=0.1)
    plt.show()
    plt.pause(0.1)
    
plt.savefig('distribution_NT1.png')

#%%
plt.close('all')
datafolder = '../../control'
files = ospath.list_files(datafolder, subfolders=True, exts='edf')
for file in tqdm(files):
    sr = SleepRecord(file)
    data = sr.raw
    mesh = specgram_multitaper(data, sr.sfreq, show_plot=False)
    plt.hist(mesh.ravel(), bins=np.arange(0,1,0.01), alpha=0.1)
    plt.show()
    plt.pause(0.1)
    
plt.savefig('distribution_control.png')