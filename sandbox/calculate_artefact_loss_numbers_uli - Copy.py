 sot# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:54:50 2020

@author: Simon
"""
import numpy as np
from sleep import SleepSet
import config as cfg
import features
import pandas as pd
import dateparser
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage.morphology import binary_dilation

def block_length(data, seconds):
    data = np.repeat(data, 30)
    seconds -= 15 # substract window middle
    if seconds>0:
        data = binary_dilation(data, structure=[True,True,True], 
                                   iterations=seconds)
    data = data.reshape([-1,30]).max(1)
    return data





if __name__=='__main__':
    ss = SleepSet(cfg.folder_unisens).stratify()
    header = ['Code',
                 'Epochs',
                 '% artefact 30s',
                 '% artefact 300s',]
    table = pd.DataFrame(columns = header)
    
    for p in tqdm(ss, 'caluclating'):
        art_30 = p.get_artefacts(only_sleeptime=True, wsize=30)
        art_300 = p.get_artefacts(only_sleeptime=True, wsize=300)
        row = {'Code': p.code,
               'Epochs': p.epochs_hypno,
               '% artefact 30s':f'{np.mean(art_30)*100:.1f}',
               '% artefact 300s':f'{np.mean(art_300)*100:.1f}'}

        table = table.append(row, ignore_index=True, sort=False) 

    table.to_excel(cfg.documents + '/artefact_corr_improvement_new.xls')
