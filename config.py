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
import os
import ospath
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
home = os.path.expanduser('~')


if username == 'nd269' and host=='ess-donatra':
    data = 'Z:/NT1-HRV/'
    share = ospath.join(home,'Dropbox/nt1-hrv-share/')
    documents = ospath.join(home,'/Dropbox/nt1-hrv-documents')
    
elif username == 'simon' and host=='desktop-tdifgpi':
    data = 'Z:/NT1-HRV/'
    nt1_1 = 'F:/01_Polysomnographien/02_Daten Hephata-Klinik Treysa (33 Patienten mit NT1)/01_Daten'
    nt1_2 = ''
    share = ospath.join(home,'Dropbox/nt1-hrv-share/')
    documents = ospath.join(home,'Dropbox/nt1-hrv-documents')
    
else:
    print('Username {} on host{}({}) has no configuration.\n'.format(username,host,system) + \
    'please set configuration in config.py')
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# USER SPECIFIC CONFIGURATION
###############################
    
root_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in vars() else ''