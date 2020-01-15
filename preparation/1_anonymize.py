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
import dateparser
import datetime
import shutil
import pandas as pd


# we take the files from nt1_1 and nt1_2, new data folders need to be added 
# here. For now, only NT1 patiens are treated here.
nt1_datafolder = ospath.join(cfg.data ,'NT1')
nt1_1 = 'Z:/02_Daten Hephata-Klinik Treysa (33 Patienten mit NT1)/01_Daten'#cfg.nt1_1
documents = cfg.documents
files = ospath.list_files(nt1_1, exts='edf')

def codify(filename): 
    number = sum([ord(c) for c in filename])
    string = str(sum([ord(c)%2 for c in filename])) + str(sum([int(x) for x in str(len(filename))]))
    return string + '_' + str(number)


if __name__ == '__main__':
    old_names = []
    new_names = []
    new_names2 = []
    for i,file in enumerate(tqdm(files)):
        # we use a coding system to create (hopefully) unique new file names
        # that do not depend on a index. this makes it easier to
        # lateron add/remove files without mixing up the indices
        filename = ospath.splitext(ospath.basename(file))[0]
        new_name = 'NT_' + codify(filename)
        new_name2 = 'NT1{:0>2}'.format(i+1)
        
        hrv = ospath.join(nt1_datafolder, new_name2 + '_hrv.mat')
        hrv_new = ospath.join(nt1_datafolder, new_name + '_hrv.mat')
        shutil.copy(hrv, hrv_new)
        new_file = ospath.join(nt1_datafolder, new_name + '.edf')
        
        old_names.append(ospath.basename(file))
        new_names.append(ospath.basename(new_file))
        new_names2.append(new_name2)
        
        if ospath.exists(new_file): 
            print ('New file extists already {}'.format(new_file))
            continue
        # assert not ospath.exists(new_file), 'New file extists already {}'.format(new_file)
        header = sleep_utils.read_edf_header(file)

        birthdate = header['birthdate']
        if birthdate!='':
            birthdate = dateparser.parse(birthdate)
            birthdate = datetime.datetime(year=birthdate.year, month=birthdate.month, day=1)
        
        # patient name will be removed and replaced
        # birthdate will be replaced with a non-identifiable version
        to_remove  = ['patientname','birthdate']
        new_values = ['xxx', birthdate]
        sleep_utils.anonymize_edf(file, new_file, verify=True)
        
        old_names.append(ospath.basename(file))
        new_names.append(ospath.basename(new_file))
        new_names2.append(new_name2)
    
    csv_mapping = ospath.join(documents, 'mapping_{}.csv'.format(\
                    ospath.basename(ospath.dirname(nt1_datafolder))))
    
    d = {'Original Name': old_names, 'Previous Name':new_names2, 'New Name':new_names}
    df = pd.DataFrame(d)
    df.to_csv(csv_mapping)
