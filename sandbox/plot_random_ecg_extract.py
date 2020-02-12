# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:08:54 2020

This file helps to easily spot files which have the wrong polarity

@author: skjerns
"""
import ospath
import config as cfg
import matplotlib.pyplot as plt
from sleep_utils import read_edf
from tqdm import tqdm
if __name__ == '__main__':
    data = cfg.data
    files = ospath.list_files(data, exts='edf') 
    _, ax = plt.subplots()
    for file in tqdm(files):
        png ='C:/Users/Simon/Desktop/seg/' + ospath.basename(file) + '.png'
        if ospath.exists(png): continue
    
        data, sig, head = read_edf(file, ch_names=['ECG I'], verbose=False)
        data = data.squeeze()
        sfreq = sig[0]['sample_rate']
        half = len(data)//2
        seg = data[half:half + 5*sfreq]
        ax.clear()
        ax.plot(seg)
        plt.savefig(png)
        