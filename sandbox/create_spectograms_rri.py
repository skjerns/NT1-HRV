# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:16:26 2020

this script creates multitaper spectrograms for
all our record files.

@author: skjerns
"""
from sleep import SleepSet
import sleep_utils
import ospath
import config as cfg
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Process, Queue

if __name__ == '__main__':  
    ss = SleepSet(cfg.folder_unisens)
    for p in tqdm(ss[600:]):
        dataset = p.get_attrib('dataset', '')
        saveas = ospath.join(cfg.documents, 'plots', p.group, dataset, p.code + '.jpg')
        if ospath.exists(saveas): continue
        p.spectogram(channels = ['ecg','RRi'], ufreq=2)
        # plt.close('all')
        break