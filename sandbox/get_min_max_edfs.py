# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:44:35 2019

@author: SimonKern
"""

from SleepData import SleepRecord
import ospath
from tqdm import tqdm
from lspopt import spectrogram_lspopt   
import numpy as np
files = ospath.list_files('C:/Users/SimonKern/Desktop/nt1-hrv/', subfolders=True, exts='edf')


mins = []
maxs = []

for file in tqdm(files):
    sr=SleepRecord(file)
    data = sr.raw
    sfreq = sr.sfreq
    perc_overlap = 1/3
    sperseg=30
    nperseg = int(round(sperseg * sfreq))
    overlap = int(round(perc_overlap * nperseg))
    freq, xy, mesh = spectrogram_lspopt(data, sfreq, nperseg=nperseg,
                                       noverlap=overlap, c_parameter=20.)
    mesh = 20 * np.log10(mesh)
    idx_notfinite = np.isfinite(mesh)==False
    mesh[idx_notfinite] = np.min(mesh[~idx_notfinite])
    mins.append(mesh.min())
    maxs.append(mesh.max())