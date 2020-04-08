# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:34:01 2020

@author: skjerns
"""
import config as cfg
import ospath
import sleep_utils
from tqdm import tqdm
import numpy as np 

folder = cfg.folder_edf
files = ospath.list_files(folder, exts=['hypno'])

hypnos = {}
for file in tqdm(files):
    hypno = sleep_utils.read_hypnogram(file)
    hypnos[ospath.basename(file)] = hypno



res = np.zeros([len(hypnos), len(hypnos)])
samehypno = set()
for i, hypno1 in enumerate(hypnos.values()):
    for j, hypno2 in enumerate(hypnos.values()):
        minlen = min(len(hypno1), len(hypno2))
        same = np.mean(hypno1[:minlen]==hypno2[:minlen])
        res[i, j] = same
        if same==1:
            name1 = list(hypnos)[i]
            name2 = list(hypnos)[j]
            if name1!=name2:
                samehypno.add(tuple(sorted((name1, name2))))
                