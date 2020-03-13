# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:11:30 2020

@author: skjerns
"""
import os, sys
import shutil
import unittest
import tempfile
from unisens import SignalEntry, Unisens, ValuesEntry
import numpy as np
from datetime import datetime, date
from sleep import Patient, SleepSet
import sleep_utils

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
    
    def test_minmax2offset(self):
        dmin = -np.random.randint(-32766, -1)
        dmax = -np.random.randint(1,  32766)
        pmin = -np.random.randint(-500, -1)
        pmax = -np.random.randint(1, 500)
        
        signal = np.random.randint(-dmin, -dmax, 100)
        signal_p1 = sleep_utils.dig2phys(signal, dmin, dmax, pmin, pmax)
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        signal_p2 = lsb*(signal + offset)
        
        np.testing.assert_allclose(signal_p1, signal_p2)

class TestPatient(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='unisens_')
        cls.patient_list = [os.path.join(cls.tmpdir, 'u1'), 
                             os.path.join(cls.tmpdir, 'u2')]
        cls.patient_str = []
        cls.patients = [Patient(f) for f in cls.patient_list]
        
        seconds = 15000
        for p in cls.patients:
            x = np.arange(256*seconds)
            sine = np.sin(2 * np.pi * 5 * x / 256)                  
            signal = SignalEntry('eeg.bin', parent=p._folder).set_data(sine, samplingRate=256)
            p.add_entry(signal)
        hypnogram = np.random.randint(0,5, [seconds//30])
        hypnogram = np.vstack([np.arange(seconds//30),hypnogram])
        cls.p = Patient(cls.tmpdir + '/test/')
        cls.p.add_entry(signal)
        hypno = ValuesEntry(id = 'hypnogram.csv', parent=cls.p)
        hypno.set_data(hypnogram, samplingRate=1/30)
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        
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
        p = self.p
        p.plot(hypnogram=True)     
        p.plot(hypnogram=False)     
        p.plot('hypnogram', hypnogram=False)  
        
        
if __name__ == '__main__':
    unittest.main()
    
