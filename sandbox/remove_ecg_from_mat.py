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
import argparse



def extract_from_mat(folder=None, overwrite=False):
    print('Converting files in ', folder)
    files = ospath.list_files(folder, exts='hrv.mat')
    for old_file in tqdm(files):
        tmp_file = tempfile.TemporaryFile().name
        new_file = os.path.splitext(old_file)[0] + '_small.mat'
        
        if os.path.exists(new_file) and not overwrite:
            print(f'{new_file} exists, no overwrite')
            continue
        
        shutil.copy(old_file, tmp_file)
        with h5py.File(tmp_file, 'r+') as fread:
            del fread['Res']['CNT']
            fread.flush()
    
        h5repack = os.path.abspath("./h5repack.exe")
        subprocess.run([h5repack, tmp_file, new_file])
        os.remove(tmp_file)



if __name__ == '__main__':
    folder = os.path.abspath(os.path.dirname(__file__))    
    parser = argparse.ArgumentParser(description='Load the visualizer for artefacts')
    parser.add_argument('-f', '--folder', type=str, default=folder,
                         help='A folder with kubios .mat files with ECG')
    parser.add_argument('-overwrite', type=bool, default=False,
                         help='Overwrite existing files?')


    args = parser.parse_args()
    folder = args.folder
    overwrite = args.overwrite
    
    
