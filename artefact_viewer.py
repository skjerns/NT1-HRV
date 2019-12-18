# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:28:49 2019

@author: Simon
"""
import os
import misc
import stimer
import numpy as np
import config
import matplotlib.pyplot as plt
import mat73 # pip install mat73
import sleep
import argparse

edf_file = config.data + 'control/A9318.edf'
mat_file  = config.share + 'A9318_hrv.mat'

interval = 30  # intervals per window in seconds 
pos = 0
nrows = 4
ncols = 4
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
#%%
   
class ECGPlotter():
    
    
    def _load(self, edf_file, mat_file=None):
        if mat_file is None:
            mat_file = edf_file[:-4] + '_hrv.mat'
            if not os.path.exists(mat_file):
                print('{} not found'.format(mat_file))
                dir = os.path.dirname(mat_file)
                mat_file = misc.choose_file(dir, exts='mat', 
                        title='Select the corresponding MAT file by Kubios')
            
        p = sleep.Patient(edf_file, channel='ECG I')
        data = p.data
        sfreq = p.sfreq    
        
        mat = mat73.loadmat(mat_file)
        rrs = mat['Res']['HRV']['Data']['T_RR'] - p.starttime
        art = mat['Res']['HRV']['TimeVar']['Artifacts']
        

        artefacts_file = edf_file[:-4] + '.npy'  
        if os.path.exists(artefacts_file):
            artefacts = np.load(artefacts_file)
        else:
            artefacts = np.repeat(art>self.threshold, repeats=2, axis=0).T
            
        self.kubios_art = np.nan_to_num(art.squeeze(), nan=99.0)
        self.mat = mat
        self.rrs = rrs.squeeze()
        self.data = data
        self.sfreq = sfreq
        self.artefacts = artefacts
        
        self.file = edf_file
        self.mat_file = mat_file
        self.artefacts_file = artefacts_file
        
        self.save()

    
    def __init__(self, edf_file, mat_file=None, pos=0, 
                 interval=30, nrows=4, ncols=4):
        self.c_okay = (1, 1, 1, 1)       # background coloring of accepted
        self.c_art = (1, 0.8, 0.4, 0.5)  # background coloring of artefact
        self.threshold = 5
        self.pos = pos
        self.interval = interval
        self.nrows = nrows
        self.ncols = ncols
        self.total = nrows*ncols
        
        self._load(edf_file=edf_file, mat_file=mat_file)
        
        # set up the plot, connect the button presses
        self.axs = []
        self.fig, self.axs = plt.subplots(nrows, ncols)
        self.axs = [item for sublist in self.axs for item in sublist]
        _ = self.fig.canvas.mpl_connect("button_press_event", self.mouse_toggle_select)
        _ = self.fig.canvas.mpl_connect("key_press_event", self.key_press)

        
        self.background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)
        self.update()
        
        
    def save(self):
        np.save(self.artefacts_file, self.artefacts)
        print('Saved artefacts to {}'.format(self.artefacts_file))
        
    def draw(self):
        self.fig.canvas.draw() 
        
    def get_rrs(self, plot_nr, plotdata):
        sec = (self.pos+plot_nr)*interval
        idx_start = np.searchsorted(self.rrs, sec)
        idx_stop  = np.searchsorted(self.rrs, sec+interval)
        rr = self.rrs[idx_start:idx_stop]*self.sfreq
        yy = self.data[rr.astype(int)]
        rr = rr-sec*self.sfreq
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
            rr, yy = self.get_rrs(i, plotdata)
            ax.plot(plotdata, linewidth=0.5)
            ax.scatter(rr, yy , marker='x', color='r', linewidth=0.5,alpha=0.7)
            ax.text(0,ax.get_ylim()[1]+50,'{:.1f}%'.format(
                    self.kubios_art[pos+i]),fontsize=8)            
            xmin, middle, xmax = self._get_xlims(ax)
            if self.artefacts[pos+i][0]:
                ax.axvspan(xmin, middle, facecolor=self.c_art, zorder=-100)
            if self.artefacts[pos+i][1]:
                ax.axvspan(middle, xmax, facecolor=self.c_art, zorder=-100)
            ax.set_xlim([xmin, xmax])
        stimer.start('draw')
        self.draw()
        stimer.stop('draw')

            
        titel = '{}/{}'.format(pos//self.total, 
                               len(data)//sfreq//interval//self.total)
        plt.suptitle(titel)
        self.draw()
        stimer.stop('update')


    def _get_xlims(self, ax):
        """
        a function to calculate the middle of an axis.
        as the axis can be negative as well we can't just
        take the half of xlim[1]
        
        :param ax: an axis element
        :returns: 
        """
        xmin, xmax = ax.get_xlim()
        middle = ((xmax-xmin)//2)+xmin
        return xmin, middle, xmax
    
    
    def key_press(self, event):
        if event.key in ('enter', 'right'):
            self.pos += self.total
        if event.key=='left':
            if self.pos>0:
                self.pos -= self.total if self.pos>=self.total else self.pos
        self.update()
        self.save()
        
        
    def toggle_artefact(self, part, idx):
        if part=='both':
            self.toggle_artefact('left', idx)
            self.toggle_artefact('right', idx)
            return
        if part=='left': i = 0
        if part=='right':  i = 1
        ax = self.axs[idx]
        xmin, middle, xmax = self._get_xlims(ax)
        art = self.artefacts[idx+self.pos][i]
        if art:
            ax.axvspan(middle if i else xmin, xmax if i else middle, 
                       facecolor=self.c_okay, zorder=-100)
            self.artefacts[idx+self.pos][i] = False
        else:
            ax.axvspan(middle if i else xmin, xmax if i else middle, 
                       facecolor=self.c_art, zorder=-100)
            self.artefacts[idx+self.pos][i] = True
        ax.set_xlim(xmin,xmax)

        
    def mouse_toggle_select(self, event):
        idx = self.axs.index(event.inaxes)
        ax = self.axs[idx]          
        stimer.start('select')
        if(str(event.button)=='MouseButton.LEFT'):
            self.toggle_artefact('both', idx)
        elif (str(event.button)=='MouseButton.RIGHT'):
            xin, middle, xmax = self._get_xlims(ax)
            
            if event.xdata>xmax//2:
                self.toggle_artefact('right', idx)
            else:
                self.toggle_artefact('left', idx)
        elif (str(event.button)=='MouseButton.MIDDLE'):
            ax.show() 
        elif (str(event.button)=='MouseButton.BACK'):
            self.fig.canvas.restore_region(self.background)
            self.fig.canvas.blit(ax.bbox)
        else:
            print('unknown key', event.button)
        plt.pause(0.001)
        stimer.stop('select')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Load the visualizer for artefacts')
    parser.add_argument('-edf', '--edf_file', type=str,
                         help='A link to an edf-file. The channel ECG I  needs to be present.')
    parser.add_argument('-mat', '--mat_file', type=str,
                         help='A link to an mat-file created by Kubios.'
                              'It contains the RRs and the artefact annotation')
    args = parser.parse_args()
    edf_file = args.edf_file
    mat_file = args.mat_file
    
    if edf_file is None:
        edf_file = misc.choose_file(exts=['edf', 'npy'], 
                                    title='Choose a EDF to display')
    print('loading {}'.format(edf_file))
    
    self = ECGPlotter(edf_file=edf_file, mat_file=mat_file)
    


