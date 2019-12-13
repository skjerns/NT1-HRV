# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:28:49 2019

@author: Simon
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sleep

file = 'Z:/NT1-HRV/control/A9318.edf'
mat  = 'C:/Users/Simon/dropbox/nt1-hrv-share/A9318_hrv.mat'
p = sleep.Patient(file, channel='ECG I')
data = p.data
interval = 30  # intervals per window in seconds 
sfreq = p.sfreq
pos = 0
nrows = 4
ncols = 4

#%%
   
class ECGPlotter():
    
    def __init__(self, data, sfreq, pos=0, interval=30, nrows=4, ncols=4):
        self.data = data
        self.sfreq = sfreq
        self.pos = pos
        self.interval = interval
        self.nrows = nrows
        self.ncols = ncols
        self.total = nrows*ncols
        
        self.axs = []
        self.fig, self.axs = plt.subplots(nrows, ncols)
        self.axs = [item for sublist in self.axs for item in sublist]
        self._y = self.fig.canvas.mpl_connect("button_press_event", self.toggle_select)
        self._x = self.fig.canvas.mpl_connect("key_press_event", self.press_go)
        self.update()
        
        
    def draw(self):
        self.fig.canvas.draw() 
        
    def update(self):
        pos = self.pos
        data = self.data
        sfreq = self.sfreq
        interval = self.interval
        for i in range(self.total):
            plotdata = data[(pos+i)*interval*sfreq:(pos+i+1)*interval*sfreq]
            ax  = self.axs[i]
            ax.set_facecolor((1,1,1,1))            
            ax.clear()
            ax.plot(plotdata)
        titel = '{}/{}'.format(pos//self.total, len(data)//sfreq//interval//self.total)
        plt.suptitle(titel)
        self.draw()
    
    def press_go(self, event):
        print(event.key)
        if event.key in ('enter', 'right'):
            self.pos += self.total
        if event.key=='left':
            if self.pos>0:
                self.pos -= self.total if self.pos>=self.total else self.pos
        self.update()
        
        
    def toggle_select(self, event):
        idx = self.axs.index(event.inaxes)
        ax = self.axs[idx]
        print ("event in ax {}".format(idx))
        if ax.get_facecolor()==(1,1,1,1):
            ax.set_facecolor((1.0, 0.47, 0.42))
        else:
            ax.set_facecolor((1,1,1,1))
        plt.show()
        self.fig.canvas.draw() 
    


self = ECGPlotter(data, p.sfreq)
plt.show()





