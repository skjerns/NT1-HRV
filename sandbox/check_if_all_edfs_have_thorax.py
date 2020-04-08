# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:50:38 2020

RES&LT: yes they all have thorax channel

@author: skjerns
"""

from pyedflib import highlevel
import ospath
from tqdm import tqdm

files = ospath.list_files('Z:/NT1-HRV-data_old', exts='edf')

for file in tqdm(files):
    channels = highlevel.read_edf_header(file)['channels']
    assert 'Effort THO' in channels or 'Thorax' in channels