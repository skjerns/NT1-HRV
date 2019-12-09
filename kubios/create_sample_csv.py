# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:17:44 2019
@author: Simon

this script can be used to create the Kubios_Samples.csv files (segment creation)
This can be used to create segments in different lengths
"""

from datetime import datetime, timedelta

def create_sample_row(filename, sample_length, distance=None):
    """
    Creates a row for the Kubios_Samples.csv.
    
    :param filename: the corresponding EDF/RR file
    :param sample_length: The length of one sample / window for HRV calculation in seconds
    :param distance: The distance between the samples in seconds. If None or 0, will be sample_length.
    :returns: a string corresponding to 
    """
    if distance==0 or distance is None:
        distance = sample_length
        
    header = 'A9318.edf, 1'
    
    s = [header]
    for i in range(300):
        start = sample_length * (i)
        end   = sample_length * (i+1)
        s.append('S{},{},{}'.format(i, start,end))
    re = ','.join(s)
