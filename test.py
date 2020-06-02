# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:11:30 2020

@author: skjerns
"""
import os, sys
import shutil
import unittest
import tempfile
import ospath
import numpy as np
import pickle
import features
from unisens import SignalEntry, ValuesEntry, EventEntry
from sleep import Patient, SleepSet
import sleep_utils
from unisens import CustomEntry
import matplotlib.pyplot as plt



class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
    
    def test_extraction(self): 
        pad = True
        sfreq = 1
        wsize = 60
        signal = np.arange(90)
        steps = 30
        windows = features.extract_windows(signal, sfreq, wsize, steps, pad)
        assert windows.shape==(3,60)
        
        
    def test_mean_HR(self):
        RR_windows = [np.ones(30), np.ones(60)/2, np.ones(60)*1.5, []]
        feat = features.mean_HR(RR_windows)
        np.testing.assert_array_equal(feat, [60, 120, 40, np.nan])

        
    def test_meanRR(self):
        RR_windows = [np.ones(30), np.ones(60)/2, np.ones(60)*1.5, []]
        feat = features.mean_RR(RR_windows)
        np.testing.assert_array_equal(feat, [1, 0.5, 1.5, np.nan])
        
        
    def test_RMSSD(self):
        RR_windows = [np.arange(1,10, 1.5), []]
        feat = features.RMSSD(RR_windows)
        np.testing.assert_array_equal(feat, [1.5, np.nan])
        
        
    def test_pNN50(self):
        RR_windows = [np.arange(1, 2, 0.04), [1, 1.05, 1.1, 1.16, 1.17, 2], np.arange(1, 2, 0.06),  []]
        feat = features.pNN50(RR_windows)
        np.testing.assert_array_equal(feat, [0, 80, 100, np.nan])
        
    def test_SDNN(self):
        RR_windows = [np.ones(30), [0, 1],  []]        
        feat = features.SDNN(RR_windows)
        np.testing.assert_array_equal(feat, [0, 0.5, np.nan])       
        
        
    def test_LF(self):
        raise NotImplementedError

    def test_HF(self):
        raise NotImplementedError

    def test_LF_HF(self):
        raise NotImplementedError



class TestUtils(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
    
    def test_minmax2offset(self):
        dmin = np.random.randint(-32766, -1)
        dmax = np.random.randint(1,  32766)
        pmin = np.random.randint(-500, -1)
        pmax = np.random.randint(1, 500)
        
        signal = np.random.randint(dmin, dmax, 100)
        signal_p1 = sleep_utils.dig2phys(signal, dmin, dmax, pmin, pmax)
        
        lsb, offset = sleep_utils.minmax2lsb(dmin, dmax, pmin, pmax)
        signal_p2 = lsb*(signal + offset)
                
        np.testing.assert_allclose(signal_p1, signal_p2)



class TestPatient(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='testpatient_')
        cls.patient_list = [os.path.join(cls.tmpdir, 'u1'), 
                             os.path.join(cls.tmpdir, 'u2')]
        cls.patient_str = []
        cls.patients = [Patient(f) for f in cls.patient_list]
        
        seconds = 15000//2
        for p in cls.patients:
            x = np.arange(32*seconds)
            sine = np.sin(2 * np.pi * 5 * x / 32)                  
            signal = SignalEntry('eeg.bin', parent=p._folder).set_data(sine, samplingRate=32)
            p.add_entry(signal)
        hypnogram = np.random.randint(0,5, [seconds//30])
        hypnogram = np.vstack([np.arange(seconds//30),hypnogram])
    
        p = Patient(cls.tmpdir + '_xxx', makenew=True)
        hypno = [0]*5 + [2]*10 + [3]*3 + [5]*4 + [5] + [2]*4 + [0]*10      
        times = np.arange(len(hypno))
        hypno = np.vstack([times, hypno]).T
        

        arousals = [[7, 5], [20, 2]]
        ecg = np.random.rand(600 * 256)
        eeg = np.random.rand(600 * 256)
        
        SignalEntry('ECG.bin', parent=p).set_data(ecg)
        SignalEntry('EEG.bin', parent=p).set_data(eeg)
        EventEntry('hypnogram.csv', parent=p).set_data(hypno, samplingRate=1/30)
        EventEntry('arousals.csv', parent=p).set_data(arousals, samplingRate=1)
        cls.p = p
        
        
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
        
    # def test_plot(self):
    #     p = self.p
    #     p.plot(hypnogram=True)
    #     p.plot(hypnogram=False)     
    #     p.plot('artefacts', hypnogram=False)
    #     self.assertTrue(os.path.isfile(ospath.join(p._folder, '/plots', 'plot_artefacts.png')))
    #     self.assertTrue(os.path.isfile(ospath.join(p._folder, '/plots', 'plot_eeg.png')))

    def test_load_sleeptime_only(self):
        
        self.p.get_hypno()
        arousals1 = self.p.get_arousals()
        ecg1 = self.p.get_ecg()
        eeg1 = self.p.get_signal('eeg')
        
        onset = self.p.sleep_onset
        offset = self.p.sleep_offset
        
        self.assertEqual(self.p.sleep_onset, 5*30)
        self.assertEqual(self.p.sleep_offset, 27*30)       
        
        hypno2 = self.p.get_hypno(only_sleeptime=True)
        arousals2 = self.p.get_arousals(only_sleeptime=True)
        ecg2 = self.p.get_ecg(only_sleeptime=True)
        eeg2 = self.p.get_signal('eeg', only_sleeptime=True)

        np.testing.assert_array_equal(hypno2, [2]*10 + [3]*3 + [5]*4 + [5] + [2]*4 )
        np.testing.assert_array_equal(arousals2, arousals1-onset//30 )
        np.testing.assert_array_equal(ecg2, ecg1[onset*256:offset*256] )
        np.testing.assert_array_equal(eeg2, eeg1[onset*256:offset*256] )

        
    def test_serialize(self)       :
        with open(self.tmpdir + '/asd.pkl', 'wb') as f:   
            pickle.dump(self.p, f)
            
        with open(self.tmpdir + '/asd.pkl', 'rb') as f:   
            pickle.load(f)         
            
          
#%% main
if __name__ == '__main__':
    plt.close('all')
    location = features.memory.location 
    features.memory.location = None
    unittest.main()
    features.memory.location = location
    
