# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 09:23:25 2024

@author: Simon
"""
import sys;sys.path.append('..')
from datetime import datetime
from edfio import read_edf
import misc
import sleepecg
import config as cfg
import matplotlib.pyplot as plt
from datetime import datetime
from edfio import read_edf
import sleepecg
from sleep import SleepSet
from tqdm import tqdm

misc.low_priority() # set low prio to not clogg CPU

plt.close('all')
ss = SleepSet(cfg.folder_unisens)
# ss = ss.stratify() # only use matched participants

p = ss[1]

# get ECG time series and sampling frequency

hypno = {}

for p in tqdm(ss):
    ecg = p.ecg.get_data().squeeze()
    fs = p.ecg.sampleRate

    # ecg = edf.get_signal("ECG").data
    # fs = edf.get_signal("ECG").sampling_frequency

    # detect heartbeats
    beats = sleepecg.detect_heartbeats(ecg, fs)
    sleepecg.plot_ecg(ecg, fs, beats=beats)

    # load SleepECG classifier (requires tensorflow)
    clf = sleepecg.load_classifier("wrn-gru-mesa", "SleepECG")

    # predict sleep stages
    record = sleepecg.SleepRecord(
        sleep_stage_duration=30,
        # recording_start_time=start,
        heartbeat_times=beats / fs,
    )

    stages = sleepecg.stage(clf, record, return_mode="str")
    stages_prob = sleepecg.stage(clf, record, return_mode="prob")

    sleepecg.plot_hypnogram(
        record,
        stages_prob,
        stages_mode=clf.stages_mode,
        merge_annotations=True,
    )

    hypno[p.code] = stages
