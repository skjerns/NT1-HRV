# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:55:26 2020

@author: skjerns
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import misc
import ospath
import os

folder = misc.choose_folder(title='Choose folder to rename')

files = ospath.list_files(folder, exts='txt')

for file in files:
    if 'Klassifizierte Arousal - ' in file:
        newfile = file.replace('Klassifizierte Arousal - ', '')
        newfile = newfile[:-4] + '_arousal.txt'
        newfile = newfile.replace(' ', '_')
    elif 'Schlafprofil - ' in file:
        newfile = file.replace('Schlafprofil - ', '')
        newfile = newfile.replace(' ', '_')
    else:
        continue

    print(f'rename {ospath.basename(file)} to {ospath.basename(newfile)}')
    os.rename(file, newfile)