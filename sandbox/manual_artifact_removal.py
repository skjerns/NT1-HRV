# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:28:49 2019

@author: Simon
"""
import stimer
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sleep
import sleep_utils


file = 'Z:/NT1-HRV/control/A9318.edf'
matfile  = 'C:/Users/Simon/dropbox/nt1-hrv-share/A9318_hrv.mat'
p = sleep.Patient(file, channel='ECG I')
mat = sleep_utils.h5py2dict(matfile)
data = p.data
interval = 30  # intervals per window in seconds 
sfreq = p.sfreq
pos = 0
nrows = 4
ncols = 4

#%%
   
class ECGPlotter():
    
    def __init__(self, data, sfreq, rrs, art, pos=0, interval=30, nrows=4, ncols=4):
        self.data = data
        self.rrs = rrs.squeeze()
        self.art = art.squeeze()
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
        
        self.background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)
        self.update()
        
    def draw(self):
        self.fig.canvas.draw() 
        
    def get_rrs(self, plot_nr, plotdata):
        sec = (self.pos+plot_nr)*interval
        idx_start = np.searchsorted(self.rrs, sec)
        idx_stop  = np.searchsorted(self.rrs, sec+interval)
        rr = self.rrs[idx_start:idx_stop]*self.sfreq
        yy = data[rr.astype(int)]
        rr = rr-sec*sfreq
        return rr, yy
    
    def update(self):
        stimer.start('update')
        pos = self.pos
        data = self.data
        sfreq = self.sfreq
        interval = self.interval
        for i in range(self.total):
            plotdata = data[(pos+i)*interval*sfreq:(pos+i+1)*interval*sfreq]
            ax  = self.axs[i]
            ax.clear()
            ax.set_facecolor((1,1,1,1))            
            if art[pos+i]>5: ax.set_facecolor((1.0, 0.47, 0.42))
            rr, yy = self.get_rrs(i, plotdata)
            ax.plot(plotdata)
            ax.scatter(rr, yy , marker='x', color='r', linewidth=0.75, alpha=0.8)
            ax.text(0,ax.get_ylim()[1]+50,'{:.1f}%'.format(art[pos+i][0]),fontsize=8)
            
        titel = '{}/{}'.format(pos//self.total, len(data)//sfreq//interval//self.total)
        plt.suptitle(titel)
        self.draw()
        stimer.stop('update')

    
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
        stimer.start('select')
        if(str(event.button)=='MouseButton.LEFT'):
            plt.pause(0.001)
        elif (str(event.button)=='MouseButton.RIGHT'):
            self.fig.canvas.draw() 
        elif (str(event.button)=='MouseButton.MIDDLE'):
            ax.show() 
        elif (str(event.button)=='MouseButton.BACK'):
            
            self.fig.canvas.restore_region(self.background)
            self.fig.canvas.blit(ax.bbox)
        else:
            print(event.button)
        stimer.stop('select')
    
rrs = mat['Res']['HRV']['Data']['T_RR'] - p.starttime
art = mat['Res']['HRV']['TimeVar']['Artifacts']


self = ECGPlotter(data, p.sfreq, rrs, art)
ax = self.axs[0]
plt.show()





