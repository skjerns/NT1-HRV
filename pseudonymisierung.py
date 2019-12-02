# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:24:17 2019

@author: SimonKern
"""
names = []
new_names = []

from tqdm import tqdm
import ospath
import tools

datafolder = 'C:/Users/SimonKern/Desktop/nc1/nc1/'
files = ospath.list_files(datafolder)

for i, file in enumerate(files):
    i = i+1
    header = tools.read_edf_header(file)
    name = header['patientname']
    n_chs = len(header['channels'])
    new_name = 'NC1{:0>2}'.format(i)
    header['patientname'] = new_name
    header['birthdate'] = ''
    new_file = ospath.join(datafolder, new_name + '.edf')
    names.append(name)
    new_names.append(new_name)
    if ospath.exists(new_file): continue

    signal_headers = []
    signals = []
    for ch_nr in tqdm(range(n_chs)):
        signal, signal_header, _ = tools.read_pyedf(file, digital=True, 
                                                    ch_nrs=ch_nr, verbose=False)
        signal_headers.append(signal_header[0])
        signals.append(signal.squeeze())
    


    tools.write_pyedf(new_file, signals, signal_headers, header,digital=True)
