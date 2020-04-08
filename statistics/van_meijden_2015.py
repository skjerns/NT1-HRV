# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:59:02 2020

This script tries to replicate the findings by van Meijden (2015), 
https://doi.org/10.1111/jsr.12253

Their major finding was
    - Heart rate was significantly higher for NT1
    - Groups did not differ in heart rate variability measures.

Participants:
    - had not used and were not using medication for narcolepsy
    - gender&age matchted (+-5years), no medical conditions, not pregnant
    - no use of hypnotics or drugs affecting HR, a history of diabetes mellitus

Calculations:
    Cardiovascular:
    - the mean HR of each epoch was calculated
    - HR, HRV, resp. freq var. for each sleep stage 
    - low frequency (LF; 0.04–0.15 Hz) power via FFT (sympathetical activity)
    - high frequency (HF; 0.15–0.4 Hz) power via FFT (vagal activity)
    - total power 0–0.4 Hz range (up = decline in sympathetic tone)
    - resp. frequency from thoracic respiratory band movements
    - (HF^RF) was calculated by 0.65 x respiratory frequency (Hz)
    - (??^RF) upper limit by 1.35 x respiratory frequency (Hz)
    - the LF^RF/HF^RF ratio was calculated
    
    PSG:
    - total sleep time, 
    - sleep efficiency, 
    - latency to sleep stage 1, -
    - (REM) latency from sleep onset and 
    - stage shift index (number of shifts between sleep stages per hour) 
    - relative durations of stages 1–3 of total sleep time. 

    - HR, LF/HF, LF/HF^RF, total power of HRV, RF for sleep stages and WASO
    - epochs of sleep following onset of nocturnal sleep and 
      prior to awakening in the morning
     
     
Epochs included:
    - Median length of each sleep stage on group level, only epochs above 
    - see paper...


Additionally:
    we labelled an epoch with ‘arousal transition’ when this epoch followed a 
    transition from SWS to sleep stage 2 or 1, and from sleep stage 2 to 1.
    Only epochs without apnoeas, leg movements (LM) and arousal transitions 
    were included in the analysis.

@author: Simon Kern
"""

import config as cfg
