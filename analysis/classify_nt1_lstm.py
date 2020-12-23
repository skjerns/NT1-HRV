# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:26:37 2020

A tryout to use LSTMs to classify NT1 from features

@author: Simon Kern
"""
import os
import random as rn
import config as cfg
from sleep import SleepSet
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import shuffle
from scipy.stats import zscore
from functions import interpolate_nans
from tqdm import tqdm
import misc

np.random.seed(0)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(0)

#%%
ss = SleepSet(cfg.folder_unisens)
ss = ss.filter(lambda x: x.duration < 60*60*11) # only less than 14 hours
ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # only less than 14 hours


#%% Load data


feat_names = ['mean_HR', 'rrHRV', 'SDNN', 'RMSSD', 'RR_range', 'SDSD', 'LF_HF','SD2_SD1' ]
feats = []
for name in tqdm(feat_names, desc='Loading features'):
    feat = np.array([p.get_feat(name, only_sleeptime=True, step=150)[:60] for p in ss])
    feat = interpolate_nans(feat)
    feat = zscore(feat)
    feats.append(feat)

data_x = np.stack(feats, axis=2)
data_y = np.array([p.group=='nt1' for p in ss])

#%% create the model
cv = StratifiedKFold(shuffle=True)
y_pred = []
y_true = []
for idx_train, idx_test in cv.split(data_x, data_y, groups=data_y):
    train_x = data_x[idx_train]
    train_y = data_y[idx_train]
    test_x = data_x[idx_test]
    test_y = data_y[idx_test]

    model = Sequential()
    model.add(LSTM(20, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(20))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=32, verbose=2)
    pred = model.predict_classes(test_x)
    y_pred.extend(pred)
    y_true.extend(test_y)
    print(f'### F1: {f1_score(test_y, pred)}')

report = classification_report(y_true, y_pred, output_dict=True)
print(classification_report(y_true, y_pred)) # once more to print
misc.save_results(report, ss=ss, clf='LSTM')
