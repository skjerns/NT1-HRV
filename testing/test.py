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
import time
import features
from unisens import SignalEntry, ValuesEntry, EventEntry
from sleep import Patient, SleepSet
import sleep_utils
import config as cfg
from unisens import CustomEntry
import matplotlib.pyplot as plt
import logging as log


class TestFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.p = Patient('xx_xxx', readonly=False)
        time.sleep(0.3)
        self.p.reset()


    @classmethod
    def tearDownClass(self):
        time.sleep(0.3)
        self.p.reset()


    def test_all(self):
        feat_names = cfg.mapping_feats
        for name in feat_names:
            if name not in features.__dict__: continue
            feat30 = self.p.get_feat(name, wsize=30, step=30, cache=False)
            feat300 =  self.p.get_feat(name, wsize=300, step=30, cache=False)
            feat60 = self.p.get_feat(name, wsize=300, step=60, cache=False)
            if np.isnan(feat30).all(): print(f'WARNING: {name} has only nan for wsize 30')
            if np.isnan(feat300).all(): print(f'WARNING: {name} has only nan for wsize 300')
            if np.isnan(feat60).all(): print(f'WARNING: {name} has only nan for step 60')


    def test_RR_range(self):
        RR_windows = [np.arange(0,11), np.ones(60)/2, np.arange(-10,11), []]
        feat = features.RR_range(RR_windows)
        np.testing.assert_array_equal(feat, [10, 0, 20, np.nan])

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
        
    def test_VLF(self):
        # VlfBand(0.0033, 0.04)
        fs = 100 # sample rate
        f = 0.001 # the frequency of the signal
        x = np.arange(fs*500) # the points on the x axis for plotting
        # compute the value (amplitude) of the sin wave at the for each sample
        y = (np.sin(2*np.pi*f * (x/fs))+2)/2

        vlf = features.VLF_power([y])
        lf = features.LF_power([y])
        hf = features.HF_power([y])

        # self.assertGreater(lf, hf)
        self.assertGreater(vlf, hf)


    def test_HF(self):
        # HfBand(0.15, 0.40)
        fs = 1000 # sample rate
        f = 0.5 # the frequency of the signal
        x = np.arange(fs*5) # the points on the x axis for plotting
        # compute the value (amplitude) of the sin wave at the for each sample
        y = (np.sin(2*np.pi*f * (x/fs))+2)/2

        vlf = features.VLF_power([y])
        lf = features.LF_power([y])
        hf = features.HF_power([y])

        # self.assertGreater(hf, vlf)
        # self.assertGreater(hf, lf)




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
        cls.f = Patient('xx_xxx')
        time.sleep(0.5)
        if os.path.exists('./'+cls.f._folder + '/feats/'):
            shutil.rmtree('./'+cls.f._folder + '/feats/')
        time.sleep(0.2)
        cls.f.reset()

        # we hallucinate an entire unisens
        p = Patient(cls.tmpdir + '/000_offset', makenew=True, autosave=True)
        p.startsec = 15
        RR = np.array([1]*16 + [0.5]*59 + [0.75]*40 + [1.5]*21)
        T_RR = np.hstack([[0] , np.cumsum(RR)])+15
        pkl = {'Data':{'T_RR':T_RR, 'RR':RR}}
        p.duration = int(T_RR[-1])
        CustomEntry('feats.pkl', parent=p).set_data(pkl)

        hypno = [0, 1, 2, 3, 4, 5, 6]
        times = np.arange(len(hypno))
        hypno = np.vstack([times, hypno]).T
        EventEntry(id='hypnogram.csv', parent=p).set_data(hypno, sampleRate=1/30, contentClass='Stage', typeLength=1)
        cls.p_offset = p
        cls.hypno=hypno

        # we hallucinate an entire unisens
        p = Patient(cls.tmpdir + '/001__fulloffset', makenew=True, autosave=True)
        p.startsec = 30
        RR = np.array([0.5]*59 + [0.75]*40 + [1.5]*21)
        T_RR = np.hstack([[0] , np.cumsum(RR)])+30
        pkl = {'Data':{'T_RR':T_RR, 'RR':RR}}
        p.duration = int(T_RR[-1])
        CustomEntry('feats.pkl', parent=p).set_data(pkl)

        hypno = [0, 1, 2, 3, 4, 5, 6]
        times = np.arange(len(hypno))
        hypno = np.vstack([times, hypno]).T
        EventEntry(id='hypnogram.csv', parent=p).set_data(hypno, sampleRate=1/30, contentClass='Stage', typeLength=1)
        cls.p_full_offset = p
        cls.hypno=hypno



    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)


    def test_offset(self):
        p = self.p_offset

        for i in range(4,8):
            p.reset()
            hypno_org = list(range(i))
            times = np.arange(len(hypno_org))
            hypno_org = np.vstack([times, hypno_org]).T
            p.hypnogram.set_data(hypno_org, sampleRate=1/30, contentClass='Stage', typeLength=1)

            hypno = p.get_hypno(cache=False)
            self.assertEqual(len(hypno),len(hypno_org))
            art = p.get_artefacts(cache=False)
            feat = p.get_feat('mean_HR', wsize=30, cache=False)
            np.testing.assert_array_almost_equal(feat[:4], [60, 120, 80, 40])

            self.assertEqual(len(hypno), len(art))
            self.assertEqual(len(hypno), len(feat))

        p = self.p_full_offset
        p.offset = 30
        p.reset()
        art = p.get_artefacts(cache=False)
        feat = p.get_feat('mean_HR', wsize=30, cache=False)

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


    def test_artefacts(self):
        f = self.f
        art = f.get_artefacts(offset=False)
        self.assertEqual(len(art), 82)
        self.assertEqual(sum(art), 22)

        art2 = f.get_artefacts(wsize=300, step=30, offset=False)
        self.assertEqual(len(art2), 82)
        self.assertEqual(sum(art2), 46)
        time.sleep(0.25)
        self.f.reset()
        time.sleep(0.1)
        self.assertFalse(os.path.exists(f._folder + '/feats'))
        self.assertFalse(os.path.exists(f._folder + '/artefacts-30-30-0.npy'))
        self.assertFalse(os.path.exists(f._folder + '/artefacts-300-30-0.npy'))


    def test_feature(self):
        f = self.f
        hr = f.get_feat('mean_HR', offset=False)

        self.assertTrue(os.path.exists(f._folder + '/feats/mean_HR-30-30-0.npy'))
        self.assertTrue(os.path.exists(f._folder + '/artefacts-30-30-0.npy'))
        np.testing.assert_allclose(f.feats.__dict__['_cache_feats/mean_HR-30-30-0.npy'], hr)
        del f.feats.__dict__['_cache_feats/mean_HR-30-30-0.npy']
        hr2 = f.get_feat('mean_HR', offset=False)
        np.testing.assert_allclose(hr, hr2)

        time.sleep(0.25)
        f.reset()
        self.assertFalse(os.path.exists(f._folder + '/feats/mean_HR-30-30-0.npy'))
        self.assertFalse(os.path.exists(f._folder + '/artefacts-30-30-0.npy'))

        files = ospath.list_files(f._folder, subfolders=True)
        self.assertEqual(len(files), 3)


    def test_RRoffset(self):
        f = self.f
        T_RR1, RR1 = f.get_RR(offset=False)
        T_RR2, RR2 = f.get_RR(offset=True)

        assert T_RR1[0]<1
        assert T_RR1[0]>=0
        assert T_RR2[0]<1
        assert T_RR2[0]>=0

        art1 = f.get_artefacts(offset=False)
        art2 = f.get_artefacts(offset=True)

        hr1 = f.get_feat('mean_HR', offset=False)
        hr2 = f.get_feat('mean_HR', offset=True)

        hr3 = f.get_feat('mean_HR', offset=True, only_sleeptime=True)
        hr4 = f.get_feat('mean_HR', offset=True, only_sleeptime=True, wsize=300, step=30)
        hr5 = f.get_feat('mean_HR', offset=True, only_sleeptime=True, wsize=300, step=300)


#%% main
if __name__ == '__main__':
    plt.close('all')
    level = log.getLogger().level
    log.getLogger().setLevel(log.DEBUG)

    unittest.main()
    
    log.getLogger().setLevel(level)