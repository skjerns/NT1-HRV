# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:24:17 2019
@author: SimonKern

This script removes all patient related information from an edf
and copies them to a new location, for n
"""
import sys
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

#######################
# Settings for datasets
#######################

target_folder = cfg.data  # leads to where the final data is stored
datasets = cfg.datasets   # contains a dictionary with a mapping of datasetname:location leading to datasets
documents = cfg.documents # contains the path to the nt1-hrv-documents folder in the dropbox

#######################
# Settings for Channel 
# Renaming
#######################
ch_mapping = cfg.channel_mapping

#%%#############
## Actual code
###############



def anonymize_and_streamline(dataset_folder, target_folder, threads=False):
    """
    This function loads the edfs of a folder and
    1. removes their birthdate and patient name
    2. renames the channels to standardized channel names
    3. saves the files in another folder with a non-identifyable 
    4. verifies that the new files have the same content as the old
    """
    to_discard = [line[0] for line in misc.read_csv(cfg.edfs_discard)]
    to_invert = [line[0] for line in misc.read_csv(cfg.edfs_invert)]

    files = ospath.list_files(dataset_folder, exts='edf', subfolders=True)
    old_names = []
    new_names = []
    for i, old_file in enumerate(tqdm(files)):
        old_name = ospath.splitext(ospath.basename(old_file))[0]
        new_name = codify(old_name)
        if new_name in new_names:
            other_old = old_name[new_names.index(new_name)]
            raise Exception('Hash collision, file is already in database. '\
                            '{},{}->{}'.format(other_old, old_name, new_name))
        elif old_name in to_discard:
            print('EDF is marked as corrupt and will be discarded')
            continue
        
        new_file = ospath.join(target_folder, new_name + '.edf')
        old_names.append(old_name)
        new_names.append(new_name)  

        if ospath.exists(new_file): 
            print ('New file extists already {}'.format(new_file))

        else:
        # anonymize
            signals, signal_headers, header = sleep_utils.read_edf(old_file, 
                                                                   digital=True,
                                                                   verbose=False)
            header['birthdate'] = ''
            header['patientname'] = 'xxx'

            for shead in signal_headers:
                ch = shead['label']
                if ch in ch_mapping:
                    ch = ch_mapping[ch]
                    shead['label'] = ch
                    

            if old_name in to_invert:
                for i,sig in enumerate(signals):
                    label = signal_headers[i]['label'].lower()
                    if label == cfg.ecg_channel.lower():
                        signals[i] = -sig

            sleep_utils.write_edf(new_file, signals, signal_headers, header, 
                                  digital=True, correct=True)
            sleep_utils.compare_edf(old_file, new_file, verbose=False, threading=False)

        # also copy additional file information ie hypnograms and kubios files
        old_dir = ospath.dirname(old_file)
        add_files = ospath.list_files(old_dir, exts=['txt', 'dat', 'mat'])
        copy_as_well = [file for file in add_files if old_name[:-2] in file]
        for add_file in copy_as_well: 
            ext = ospath.splitext(add_file)[-1]
            new_additional_file = ospath.join(target_folder, new_name + ext)
            if ospath.exists(new_additional_file):continue
            try:
                shutil.copy(add_file, new_additional_file)
            except Exception as e:
                print(e)
            
    return old_names, new_names
        
if __name__ == '__main__':
    print('running in parallel. if you don\'t see output, start with python.exe')
    results = Parallel(n_jobs=4, backend='loky')\
    (delayed(anonymize_and_streamline)(dataset, target_folder=target_folder) for dataset in datasets.values())
    i=0
    for old_names, new_names in results:
        csv_file = ospath.join(documents, 'mapping_{}.csv'.format(list(datasets.keys())[i]))
        csv = pd.DataFrame(zip(old_names, new_names))
        csv.to_csv(csv_file, header=None, index=False, sep=';')
        i+=1
        
    # save as well in one large file
    all_old_names = []
    all_new_names = []
    for old_names, new_names in results:
        all_old_names.extend(old_names)
        all_new_names.extend(new_names)
    csv_file = ospath.join(documents, 'mapping_all.csv')
    csv = pd.DataFrame(zip(all_old_names, all_new_names))
    csv.to_csv(csv_file, header=None, index=False, sep=';')

        
        
        
        
    