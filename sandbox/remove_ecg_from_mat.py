# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:02:13 2020

@author: skjerns
"""
import os
import ospath
import h5py
import subprocess
import tempfile
import shutil
from tqdm import tqdm

folder = 'C:/Users/Simon/Dropbox/nt1-hrv-share/'

if __name__ == '__main__':
    files = ospath.list_files(folder, exts='hrv.mat')
    for old_file in tqdm(files):
        tmp_file = tempfile.TemporaryFile().name
        new_file = os.path.splitext(old_file)[0] + '_small.mat'
        shutil.copy(old_file, tmp_file)
        with h5py.File(tmp_file, 'r+') as fread:
            del fread['Res']['CNT']
            fread.flush()
    
        h5repack = os.path.abspath("./h5repack.exe")
        subprocess.run([h5repack, tmp_file, new_file])
        os.remove(tmp_file)
