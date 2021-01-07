# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:39:19 2021

@author: Simon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:26:37 2020

A tryout to use LSTMs to classify NT1 from features

@author: Simon Kern
"""
import os
import misc
import joblib
import numpy as np
import random as rn
import config as cfg
import tensorflow as tf
from sleep import SleepSet, Patient
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, TimeDistributed
from keras.layers import MaxPool1D, Flatten, Input

from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from functions import resample
from sklearn.utils import shuffle
from scipy.stats import zscore
from functions import interpolate_nans
from tqdm import tqdm

np.random.seed(0)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(0)

memory = joblib.Memory('z:/cache/')

@memory.cache
def cache_resampled_zscore_patient(folder, min_sfreq):
    """helper function to cache results for faster loading"""
    p = Patient(folder)
    orig_sfreq = p.ecg.sampleRate
    ecg = p.get_ecg(only_sleeptime=True)
    ecg = zscore(resample(ecg, orig_sfreq, min_sfreq))
    return ecg
#%%
ss = SleepSet(cfg.folder_unisens)

ss = ss.filter(lambda x: x.duration < 60*60*11) # only less than 14 hours
ss = ss.filter(lambda x: x.group in ['control', 'nt1']) # only less than 14 hours
ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True))<0.25) #only take patients with artefact percentage <25%



#%% Load data
min_sfreq = min([p.ecg.sampleRate for p in ss])
length = 60 * 90 # load first 1.5 hours of recording
blocksize = min_sfreq * 60 # 1 minute blocks
n_blocks = length//60

data_x = []
data_y = []
for p in tqdm(ss):
    ecg = cache_resampled_zscore_patient(p._folder, min_sfreq)[:length*min_sfreq]
    ecg = ecg.reshape([-1, blocksize])
    data_x.append(ecg)
    data_y.append(p.group=='nt1')


data_x = np.array(data_x)
data_y = np.array(data_y)

#%% create the model
cv = StratifiedKFold(shuffle=True)
y_pred = []
y_true = []
for idx_train, idx_test in cv.split(data_x, data_y, groups=data_y):
    train_x = data_x[idx_train]
    train_y = data_y[idx_train]
    test_x = data_x[idx_test]
    test_y = data_y[idx_test]

    input_layer = Input(shape=(n_blocks, blocksize, 1))

    # first create the CNN
    cnn = Sequential()
    input_shape = (blocksize, 1)
    cnn.add(Conv1D(32, 5, input_shape=input_shape))
    cnn.add(MaxPool1D(3))
    cnn.add(Conv1D(32, 5))
    cnn.add(MaxPool1D(2))
    cnn.add(Conv1D(64, 5))
    cnn.add(MaxPool1D(2))
    cnn.add(Conv1D(64, 5))
    cnn.add(MaxPool1D(2))
    cnn.add(Conv1D(64, 5))
    cnn.add(MaxPool1D(2))
    cnn.add(Conv1D(64, 5))
    cnn.add(MaxPool1D(2))
    cnn.add(Conv1D(64, 5))
    cnn.add(MaxPool1D(2))
    cnn.add(Flatten())
    cnn.add(Dense(100))
    print(cnn.output_shape)

    model = Sequential()
    model.add(input_layer)
    model.add(TimeDistributed(model))
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
misc.save_results(report, name='CNN-LSTM-raw', ss=ss, clf=model.summary())
