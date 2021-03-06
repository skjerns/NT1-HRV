# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:02:13 2020

@author: skjerns
"""
import misc
import os
import ospath
import h5py
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import argparse
import time


def extract_from_mat(file, overwrite=False):
    h5_repack = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'h5repack.exe')

    tmp_file = tempfile.TemporaryFile(prefix='remove_ecg_from_mat').name
    new_file = os.path.splitext(file)[0] + '_small.mat'
    new_file.replace('hrv_', '')


    if os.path.exists(new_file) and not overwrite:
        print(f'{new_file} exists, no overwrite')
        return
    
    shutil.copy(file, tmp_file)
    with h5py.File(tmp_file, 'r+') as fread:

        del fread['Res']['CNT']
        fread.flush()

    subprocess.run([h5_repack, tmp_file, new_file])
    os.remove(tmp_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load the visualizer for artefacts')
    parser.add_argument('-f', '--folder', type=str, default=None,
                         help='A folder with kubios .mat files with ECG')
    parser.add_argument('-overwrite', type=bool, default=False,
                         help='Overwrite existing files?')


    args = parser.parse_args()
    folder = args.folder
    overwrite = args.overwrite
    if folder is None:
        folder = misc.choose_folder('Choose a folder')

    files = [x for x in ospath.list_files(folder, exts='.mat') if not x.endswith('small.mat')]
    print('Converting files in ', folder)
    for file in tqdm(files):
        extract_from_mat(file, overwrite)
    print('\nfinished...')
    time.sleep(15)