# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:12:21 2020

in this file I check whether the rules I infer for the offset calculation
of the signals, ie. the alignment of hypnogram and signals/features are correct


@author: skjerns
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pyedflib import highlevel
import ospath
import sleep_utils
from sleep import SleepSet
import config as cfg
from datetime import datetime, timedelta

if __name__=='__main__':



    files = ospath.list_files(cfg.folder_edf, exts='edf')
    for file in files:
        hypno = file[:-4]+'.txt'

        if not ospath.exists(hypno): continue

        with open(file, 'rb') as f:
            c = f.read(184)[179:].decode().split('.')
            edf_start = datetime(2020,10,10,10, int(c[0]), int(c[1]))
            edf_startsec = edf_start.minute*60 + edf_start.second
            edf_reclen = int(f.read(60)[50:].decode())
            edf_end = edf_start + timedelta(seconds=edf_reclen)



        with open(hypno, 'r') as f:
            c = f.readlines()
            hypno_start = datetime(2020,10,10,10, int(c[7][3:5]), int(c[7][6:8]))
            hypno_startsec = hypno_start.minute*60 + hypno_start.second

            hypno_end = datetime(2020,10,10,10, int(c[-1][3:5]), int(c[-1][6:8]))
        hypno = sleep_utils.read_hypnogram(hypno)

        # print(hypno_start, edf_start)

        offset_begin = edf_startsec//30*30 - edf_startsec
        offset_end = -(edf_end.second-edf_end.second//30*30)

        if edf_start.second in [0,30]:
            # we expect the recording to start at (edf_start//30)*30
            assert offset_begin == 0

        if edf_start.second not in [0, 30]:
            assert offset_begin < 0

        if edf_end.second in [0, 30]:
            assert offset_end == 0

        if edf_end.second not in [0, 30]:
            assert offset_end < 0

        n_epochs = (edf_reclen - offset_begin + offset_end)/30
        if not n_epochs==len(hypno):
            print(file, hypno_end, edf_end)
