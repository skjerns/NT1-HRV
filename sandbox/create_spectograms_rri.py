# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:16:26 2020

this script creates multitaper spectrograms for
all our record files.

@author: skjerns
"""
import os
from sleep import SleepSet
import sleep_utils
import numpy as np
import ospath
import config as cfg
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Process, Queue

if __name__ == '__main__':  
    ss = SleepSet(cfg.folder_unisens)
    ss = ss.filter(lambda x: x.duration < 60*60*11) # only less than 14 hours
    ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # only less than 14 hours
    ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True))<0.25) #only take patients with artefact percentage <25%

    for p in tqdm(ss[:250]):
        dataset = p.get_attrib('dataset', 'none')
        saveas = ospath.join(cfg.documents, 'plots', p.group, dataset, p.code + '.jpg')
        if ospath.exists(saveas): continue
        p.spectogram(channels = ['ecg', 'RRi'], ufreq=2)
        os.makedirs(os.path.dirname(saveas), exist_ok=True)
        plt.savefig(saveas)
        plt.close('all')
