# -*- coding: utf-8 -*-
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
 'epochs artefact',
 'Uli gain',
 '0-corr 30 sec',
 '1-corr 30 sec',
 '2-corr 30 sec',
 '3-corr 30 sec',
 '300 sec loss',
 '0-corr 300 sec win',
 '1-corr 300 sec win',
 '2-corr 300 sec win',
 '3-corr 300 sec win',
 '% loss 300 sec',
 '% loss after 0-corr 300 sec',
 '% loss after 1-corr 300 sec',
 '% loss after 2-corr 300 sec',
 '% loss after 3-corr 300 sec']
    table = pd.DataFrame(columns = header)
    
    for p in ss:
        art = p.get_artefacts(only_sleeptime=True)

        kubios = p.feats.get_data()
        starttime = datetime.strptime(p.timestampStart, '%Y-%m-%dT%H:%M:%S')
        startsec = (starttime.hour * 60 + starttime.minute) * 60 + starttime.second
        p.get_hypno()
        epoch_sonset = p.sleep_onset//30
        epoch_soffset = p.sleep_offset//30
        RRi = kubios['Data']['RR']
        T_RR = kubios['Data']['T_RR'] - startsec
        RR  = np.diff(T_RR)
        t_interpolated = T_RR[:-1][RR!=RRi]
        
        corr = features.extract_RR_windows(t_interpolated, RR[RR!=RRi], wsize=30, pad=True, expected_nwin=len(art))[epoch_sonset:epoch_soffset]

        if art.mean()==0: 
            art = np.array([len(c)>0 for c in corr])
            continue
        corr = [len(c) for c in corr]

        uli_gain = np.array([a==False if c>1 else False for c, a in zip(corr,art)])
        corr = np.array([c if a else -1 for c, a in zip(corr,art)])
        
        if len(corr)!=len(art):print(f'Art!=Corr for {p.code}, {len(corr)}!={len(art)}')
        
        corr_all = corr>=0
        corr_0 = corr==0
        corr_1 = corr==1
        corr_2 = corr==2 
        corr_3 = corr==3
        
        corr_all_300 = block_length(corr_all, 300)
        corr_0_300 = block_length(corr>0, 300)
        corr_1_300 = block_length(corr>1, 300)
        corr_2_300 = block_length(corr>2, 300)
        corr_3_300 = block_length(corr>3, 300)
        
        # row = {head:None for head in header}
        row = {}
        row['Code'] = p.code
        row['Epochs'] = len(art)
        
        row['epochs artefact'] = np.sum(corr_all)
        row['Uli gain'] = np.sum(uli_gain)
        row['0-corr 30 sec'] =  np.sum(corr_0)
        row['1-corr 30 sec'] =  np.sum(corr_1)
        row['2-corr 30 sec'] =  np.sum(corr_2)
        row['3-corr 30 sec'] =  np.sum(corr_3)
        
        row['300 sec loss'] = np.sum(corr_all_300)
        row['0-corr 300 sec win'] = sum(corr_all_300) - np.sum(corr_0_300)
        row['1-corr 300 sec win'] =  np.sum(corr_0_300) - np.sum(corr_1_300)
        row['2-corr 300 sec win'] =  np.sum(corr_1_300) - np.sum(corr_2_300)
        row['3-corr 300 sec win'] =  np.sum(corr_1_300) - np.sum(corr_3_300)
        
        row['% loss 300 sec'] = f'{np.mean(corr_all_300)*100:.1f} %'
        row['% loss after 0-corr 300 sec'] = f'{np.logical_and(corr_all_300, corr_0_300).mean()*100:.1f} %'
        row['% loss after 1-corr 300 sec'] = f'{np.logical_and(corr_all_300, corr_1_300).mean()*100:.1f} %'
        row['% loss after 2-corr 300 sec'] = f'{np.logical_and(corr_all_300, corr_2_300).mean()*100:.1f} %'
        row['% loss after 3-corr 300 sec'] = f'{np.logical_and(corr_all_300, corr_3_300).mean()*100:.1f} %'

        table = table.append(row, ignore_index=True, sort=False) 

    table.to_excel(cfg.documents + '/artefact_corr_improvement.xls')
