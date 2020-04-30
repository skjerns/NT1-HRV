# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:24:17 2019
@author: SimonKern

This script removes all patient related information from an edf
and copies them to a new location, for n
"""
import sys,os
sys.path.append("..") # append to get access to upper level modules
from tqdm import tqdm
import config as cfg# here user specific configuration is saved
import ospath
import sleep_utils
import shutil
import pandas as pd
import misc
from misc import codify
from joblib import delayed, Parallel
import tempfile

#######################
# Settings for datasets
#######################

target_folder = cfg.folder_edf  # leads to where the final data is stored
datasets = cfg.datasets   # contains a dictionary with a mapping of datasetname:location leading to datasets
documents = cfg.documents # contains the path to the nt1-hrv-documents folder in the dropbox

#######################
# Settings for Channel 
# Renaming
#######################
ch_mapping = cfg.mapping_channels

###############
#%%anonymize###
###############
def anonymize_and_streamline(old_file, target_folder):
    """
    This function loads the edfs of a folder and
    1. removes their birthdate and patient name
    2. renames the channels to standardized channel names
    3. saves the files in another folder with a non-identifyable 
    4. verifies that the new files have the same content as the old
    """
    # load the two csvs with the edfs that we dont process and where the ECG is upside down
    to_discard = [line[0] for line in misc.read_csv(cfg.edfs_discard) if line[2]=='1']
    to_invert = [line[0] for line in misc.read_csv(cfg.edfs_invert)]

    # Here we read the list of controls and patients with their age and gender
    mappings = misc.read_csv(cfg.controls)
    mappings.extend(misc.read_csv(cfg.patients))
    mappings = dict([[name, {'gender':gender, 'age':age}] for name, gender, age in mappings])

    # old name is the personalized file without file extension, e.g. thomas_smith(1)
    old_name = ospath.splitext(ospath.basename(old_file))[0]
    # new name is the codified version without extension e.g '123_45678'
    new_name = codify(old_name)
    
    # use a temporary file to write and then move it, 
    # this avoids half-written files that cannot be read later
    tmp_name = tempfile.TemporaryFile().name

    if old_name in to_discard:
        print('EDF is marked as corrupt and will be discarded')
        return
    
    # this is where the anonymized file will be stored
    new_file = ospath.join(target_folder, new_name + '.edf')


    if ospath.exists(new_file): 
        print ('New file extists already {}'.format(new_file))

    else:
        # anonymize
        print ('Writing {} from {}'.format(new_file, old_name))
        assert ospath.isfile(old_file), f'{old_file} does not exist'
        signals, signal_headers, header = sleep_utils.read_edf(old_file, 
                                                               digital=True,
                                                               verbose=False)
        # remove patient info
        header['birthdate'] = ''
        header['patientname'] = new_name
        header['patientcode'] = new_name
        header['gender'] = mappings[old_name]['gender']
        header['age'] = mappings[old_name]['age']

        # rename channels to a unified notation, e.g. EKG becomes ECG I
        for shead in signal_headers:
            ch = shead['label']
            if ch in ch_mapping:
                ch = ch_mapping[ch]
                shead['label'] = ch
                
        # Invert the ECG channel if necessary
        if old_name in to_invert:
            for i,sig in enumerate(signals):
                label = signal_headers[i]['label'].lower()
                if label == cfg.ecg_channel.lower():
                    signals[i] = -sig
        
        # we write to tmp to prevent that corrupted files are not left
        print ('Writing tmp for {}'.format(new_file))
        sleep_utils.write_edf(tmp_name, signals, signal_headers, header, 
                              digital=True, correct=True)
        
        # verify that contents for both files match exactly
        print ('Verifying tmp for {}'.format(new_file))
        # embarrasing hack, as dmin/dmax dont in this files after inverting
        if not old_name=='B0036': 
            sleep_utils.compare_edf(old_file, tmp_name, verbose=False)
        
        # now we move the tmp file to its new location.
        shutil.move(tmp_name, new_file)

    # also copy additional file information ie hypnograms and kubios files
    old_dir = ospath.dirname(old_file)
    old_name = old_name.replace('_m', '').replace('_w', '') # remove gender from weitere nt1 patients
    add_files = ospath.list_files(old_dir, patterns=[f'{old_name}*txt', f'{old_name}*dat', f'{old_name}*mat'])
    for add_file in add_files: 
        # e.g. .mat or .npy etc etc
        new_add_file = ospath.join(target_folder, 
                                   ospath.basename(add_file.replace(old_name, new_name)))
        if ospath.exists(new_add_file):continue
        # hypnograms will be copied to .hypno
        try:
            new_add_file = new_add_file.replace('-Schlafprofil', '')
            new_add_file = new_add_file.replace('_sl','')
            new_add_file = new_add_file.replace('.txt', '.hypno').replace('.dat', '.hypno')
            shutil.copy(add_file, new_add_file)
        except Exception as e:
            print(e)
    return old_name, new_name




#%% Main
if __name__ == '__main__':
    print('running in parallel. if you don\'t see output, start with python.exe')
    
    # first get all edfs in all dataset folders
    files = [];  # cheeky workaround for not functioning list comprehension .extend
    _ = [files.extend(ospath.list_files(folder, exts='edf', subfolders=True)) for folder in datasets.values()]
    
    
    results = Parallel(n_jobs=4, backend='loky')(delayed(
              anonymize_and_streamline)(file, target_folder=target_folder) for file in tqdm(files, desc='processing edfs'))
    
    # remove discarded files
    results = [res for res in results if not res is None]
    
    # check for hash collision
    assert len(set(list(zip(*results))[1]))==len(list(zip(*results))[1]),\
         'ERROR: Hash collision! Check thoroughly.'
    
    csv_file = ospath.join(documents, 'mapping_all.csv')
    pd.DataFrame(results).to_csv(csv_file, header=None, index=False, sep=';')

        
        
        
        
    