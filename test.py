# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:11:30 2020

@author: skjerns
"""
import os, sys
import shutil
import unittest
import tempfile
from unisens import SignalEntry, Unisens
import numpy as np
from datetime import datetime, date
from sleep import Patient, SleepSet


class TestPatient(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='unisens_')
        self.patient_list = [os.path.join(self.tmpdir, 'u1'), 
                             os.path.join(self.tmpdir, 'u2')]
        self.patient_str = []
        self.patients = [Patient(f) for f in self.patient_list]
        
        for p in self.patients:
            x = np.arange(256*3000)
            sine = np.sin(2 * np.pi * 5 * x / 256)                  
            signal = SignalEntry('eeg.bin', folder=p._folder).set_data(sine, samplingRate=256)
            p.add_entry(signal)
            
    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        
    def test_create(self):
        sleepset1 = SleepSet(self.patient_list)
        sleepset2 = SleepSet(self.patients)
        
        assert len(sleepset1)==2
        assert len(sleepset2)==2
        
    def test_filter(self):
        s = SleepSet(self.patients)
        f = lambda x: 'u2' in x._folder
        a = s.filter(f)
        self.assertTrue(len(a)==1)
        self.assertTrue(isinstance(a, SleepSet))
        self.assertTrue('u2' in a[0]._folder)
        
    def test_plot(self):
        p = self.patients[0]
        p.plot()     
        p.plot()     

if __name__ == '__main__':
    unittest.main()
