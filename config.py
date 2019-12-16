# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:05:48 2019

this modiule can be used to store machine/user-defined variables
such as the path for data storage.

Usage eg:
    import config as cfg
    data_path = config.data_path
    
@author: skjerns
"""
import ospath
import numpy as np
import getpass
import platform

class AttrDict(dict):
    """
    A dictionary that allows access via attributes
    a['entry'] == a.entry
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GENERAL CONFIGURATION
###############################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.

if username == 'nd269' and host=='ess-donatra':
    data = 'Z:/NT1-HRV/'
    share = 'C:/Users/nd269/Dropbox/nt1-hrv-share/'
    documents = 'C:/Users/nd269/Dropbox/nt1-hrv-documents'
elif username == 'Simon':
    data = ''
else:
    'Username {} on host{}({}) has no configuration.\n' + \
    'please set configuration in config.py'
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# USER SPECIFIC CONFIGURATION
###############################
    
root_dir = ospath.abspath(ospath.dirname(__file__)) if '__file__' in vars() else ''