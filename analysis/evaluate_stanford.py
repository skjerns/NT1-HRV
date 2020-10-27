# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:57:00 2020

Evaluate Stanford-Stages-Narcolepsy detection

@author: Simon Kern
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, cohen_kappa_score
from sleep import SleepSet
import seaborn as sns
import pandas as pd

data_dir = 'Z:/NT1-HRV-unisens'
pred_dir = 'Z:/stanford'

ss_predictions = SleepSet(pred_dir)
ss_data = SleepSet(data_dir)
f1s = []
accs = []
cohens = []
nt1 = []
nt1_pred = []

for p_pred in tqdm(ss_predictions, desc='evaluating'):
    code = os.path.basename(os.path.dirname(p_pred._folder + '/'))
    p = ss_data[code]

    nt1_pred.append(p_pred.stanford_prediction)
    hypno_pred = list(zip(*p_pred.hypnogram_stanford_csv.get_data()))[1]
    hypno_pred = np.array(hypno_pred)
    hypno_pred[hypno_pred==5]=4
    hypno_pred = hypno_pred[::2]
    

    hypno = p.get_hypno()
    print(len(hypno_pred)-len(hypno))
    nt1.append(p.group=='nt1')
    minlen = min(len(hypno_pred), len(hypno))

    f1 = f1_score(hypno_pred[:minlen], hypno[:minlen], average='macro')
    acc = (np.mean(hypno_pred[:minlen]==hypno[:minlen]))
    cohen = cohen_kappa_score(hypno_pred[:minlen], hypno[:minlen])
    f1s.append(f1)
    accs.append(acc)
    cohens.append(cohen)

# nt1_pred = np.array(nt1_pred)
# nt1_pred = nt1_pred-nt1_pred.min()
# nt1_pred = nt1_pred/nt1_pred.max()
fpr, tpr, thrs = roc_curve(nt1, nt1_pred, pos_label=True)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)')
plt.title('ROC of Stanford Narcolepsy Detection (n=230)')
plt.plot([0, 1], [0, 1],'r--', c='gray')
plt.figure()
plt.title('NC1 scores')
plt.scatter(nt1, nt1_pred, marker='x')
plt.hlines(-0.03, -0.05,1.05, linestyle= 'dashed')
plt.ylabel('NC1 score')
plt.xticks([0,1],['Controls', 'NC1'])

plt.figure()
arr = np.hstack([np.vstack([f1s, ['f1']*len(f1s)]),np.vstack([cohens, ['Kappa']*len(f1s)])])
df = pd.DataFrame()
df['score'] = np.hstack([f1s, cohens])
df['group'] = ['F1']*len(f1s) + ['Kappa']*len(cohens)

plt.figure()
sns.barplot(data=df, x='group', y='score', ci='sd')
plt.title('Inter-Rater reliability')
# plt.plot(cohens)
#