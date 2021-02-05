# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:26:37 2020

A tryout to use LSTMs to classify NT1 from features

@author: Simon Kern
"""
import os
import matplotlib.pyplot as plt
import random as rn
import config as cfg
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sleep import SleepSet
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from scipy.signal import resample
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
ss = ss.filter(lambda x: np.mean(x.get_artefacts(only_sleeptime=True))<0.25) #only take patients with artefact percentage <25%


#%% Load data
length = 2 * 60 * 3 # first 3 hours
downsample = 2 # downsample factor

feat_names = ['mean_HR', 'rrHRV', 'SDNN', 'RMSSD', 'RR_range', 'SDSD','LF_power',
              'HF_power',  'LF_HF', 'SD2_SD1', 'SD1', 'SD2','SampEn',
              'pNN50', 'PetrosianFract', 'SDSD']
feats = []
for name in tqdm(feat_names, desc='Loading features'):
    feat = np.array([p.get_feat(name, only_sleeptime=True, wsize=300, step=30)[:length] for p in ss])
    feat = interpolate_nans(feat)
    feat = resample(feat, feat.shape[-1]//downsample, axis=1, window='hamming')
    # feat = interpolate_nans(zscore(feat, axis=1))
    feats.append(feat)

data_x = np.stack(feats, axis=2)
data_y = np.array([p.group=='nt1' for p in ss])

#%%
def get_lstm(n_layers, n_neurons, dropout=None):
    model = Sequential()
    for i in range(n_layers):
        model.add(LSTM(10, return_sequences= i!=n_layers-1))
        if dropout: model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])
    return model

#%% create the model
cv = StratifiedKFold(3, shuffle=True)
y_pred = []
y_true = []
hists = []

for idx_train, idx_test in cv.split(data_x, data_y, groups=data_y):
    train_x = data_x[idx_train]
    train_y = data_y[idx_train]
    test_x = data_x[idx_test]
    test_y = data_y[idx_test]

    model = get_lstm(n_layers=2, n_neurons=20, dropout=0.2)

    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=32, verbose=2)

    history = model.history.history
    hists.append(history)

    pred = model.predict_classes(test_x)
    y_pred.extend(pred)
    y_true.extend(test_y)
    print(f'### F1: {f1_score(test_y, pred)}')


keys = list(hists[0].keys())
hists = {key:np.mean([hist[key] for hist in hists], 0) for key in keys}
_, axs = plt.subplots(2,2)
for i, metric in enumerate(['loss', 'precision', 'recall']):
    ax = axs.flatten()[i]
    ax.plot(hists[metric])
    ax.plot(hists[f'val_{metric}'])
    ax.legend([metric, f'val_{metric}'])
    ax.set_title(metric)
plt.show()
plt.pause(0.01)

report = classification_report(y_true, y_pred, output_dict=True)
print(classification_report(y_true, y_pred)) # once more to print
misc.save_results(report, name='LSTM-feat', ss=ss, clf=model.summary())
