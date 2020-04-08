# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:48:42 2020

@author: skjerns
"""
import config as cfg
import ospath
import shutil
import sleep_utils
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


files = ospath.list_files(cfg.folder_edf, exts=['hypno'])

accuracy = []
kohen = []
a=[]
b=[]


for file in files:
    if ospath.exists(file.replace('.hypno', '.txt')):
        hypno1 = sleep_utils.read_hypnogram(file)
        hypno2 = sleep_utils.read_hypnogram(file.replace('.hypno', '.txt'))
        minlen = min(len(hypno1), len(hypno2))
        hypno1 = hypno1[:minlen]
        hypno2 = hypno2[:minlen]
        accuracy.append(np.mean(hypno1==hypno2))
        kohen.append(cohen_kappa_score(hypno1, hypno2))
        
        hypno1[0]=5

        labels = {0: 'W', 4: 'REM', 1: 'S1', 2: 'S2', 3: 'SWS', 5: 'A'}

        if accuracy[-1]>0.65:continue
        a.append(accuracy[-1])
        b.append(kohen[-1])

        fig, axs = plt.subplots(2,1)
        sleep_utils.plot_hypnogram(hypno1, ax=axs[0], labeldict=labels)
        sleep_utils.plot_hypnogram(hypno2, ax=axs[1], labeldict=labels)
        plt.title('Second rater')
        plt.suptitle(f'{file} acc {accuracy[-1]:.2f}, kohen {kohen[-1]:.2f}, {ospath.basename(file)}')