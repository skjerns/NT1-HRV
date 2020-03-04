# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:38:50 2020

tests for functions

@author: skjerns
"""
import sleep_utils
import os
import unittest
import tempfile
import numpy as np


class Testing(unittest.TestCase):
    
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
        
if __name__ == '__main__':
    unittest.main()



