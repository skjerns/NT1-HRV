# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:28:49 2019

@author: Simon
"""
import os
import misc
import numpy as np
import matplotlib.pyplot as plt
import mat73 # pip install mat73
import sleep
import argparse
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.INFO)
#%%
   


class ECGPlotter():
    
    def _handle_close(self,evt):
        self.save()
        plt.close('all')
        return
    
    def detect_flatline(self):
        logging.info('Detecting Artefacts')
        data = self.data.copy().squeeze()
        data = data[:len(data)-len(data)%(self.sfreq*self.interval)]
        data = data.reshape([-1 ,self.sfreq*self.interval//2])
        flat = np.mean(np.logical_and(data<2,data>-2),-1)
        flat = flat>0.1
        flat.resize(self.artefacts.shape)
        self.flat=flat
        self.artefacts[flat]=True
        return True
    
    def _load(self, edf_file, mat_file=None):
        if mat_file is None:
            mat_file = edf_file[:-4] + '_hrv.mat'
            if not os.path.exists(mat_file):
                print('{} not found'.format(mat_file))
                dir = os.path.dirname(mat_file)
                mat_file = misc.choose_file(dir, exts='mat', 
                        title='Select the corresponding MAT file by Kubios')
            
        p = sleep.Patient(edf_file, channel='ECG')
        data = p.data
        sfreq = p.sfreq    
        self.data = data
        self.sfreq = sfreq
        
        try:
            mat = mat73.loadmat(mat_file, verbose=False)
            rrs = mat['Res']['HRV']['Data']['T_RR'] - p.starttime
            art = mat['Res']['HRV']['TimeVar']['Artifacts']
        except:
            logging.error('Mat file not found.')            

        artefacts_file = edf_file[:-4] + '.npy'  
        if os.path.exists(artefacts_file):
            self.artefacts = np.load(artefacts_file)
        else:
            art = np.nan_to_num(art, nan=99)
            self.artefacts = np.repeat(art>self.threshold, repeats=2, axis=0).T
            self.detect_flatline()
            
        self.kubios_art = np.nan_to_num(art.squeeze(), nan=99.0)
        self.mat = mat
        self.rrs = rrs.squeeze()

        
        self.file = edf_file
        self.mat_file = mat_file
        self.artefacts_file = artefacts_file
        self.max_page = len(data)//sfreq//self.interval//self.gridsize
        
        self.save()

    
    def __init__(self, edf_file, mat_file=None, page=0, 
                 interval=30, nrows=4, ncols=4):
        self.c_okay = (1, 1, 1, 1)       # background coloring of accepted
        self.c_art = (1, 0.8, 0.4, 0.5)  # background coloring of artefact
        self.threshold = 5
        self.page = page
        self.interval = interval
        self.nrows = nrows
        self.ncols = ncols
        self.gridsize = nrows*ncols
        
        self._load(edf_file=edf_file, mat_file=mat_file)
        # set up the plot, connect the button presses
        self.axs = []
        self.fig, self.axs = plt.subplots(nrows, ncols)
        self.axs = [item for sublist in self.axs for item in sublist]
        _ = self.fig.canvas.mpl_connect("button_press_event", self.mouse_toggle_select)
        _ = self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        _ = self.fig.canvas.mpl_connect('close_event', self._handle_close)
        self.background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)
        self.update()
        
        
        
        # sanity check
        epochs = len(self.data)/self.sfreq/30
        if epochs!=self.artefacts.shape[0]:
            print('WARNING: {} epochs, but {} kubios annotations?'.format(
                  epochs, self.artefacts.shape[0]))
        
    def save(self):
        np.save(self.artefacts_file, self.artefacts)
        print('Saved artefacts to {}'.format(self.artefacts_file))
        
    def draw(self):
        self.fig.canvas.draw() 
        
    def get_rrs(self, plot_nr, plotdata):
        sec = (self.page*self.gridsize+plot_nr)*self.interval
        idx_start = np.searchsorted(self.rrs, sec)
        idx_stop  = np.searchsorted(self.rrs, sec+self.interval)
        rr = (self.rrs[idx_start:idx_stop]*self.sfreq)
        yy = self.data[rr.round().astype(int)]
        rr = rr-sec*self.sfreq
        return rr, yy
    
    def update(self):
        gridsize = self.gridsize
        page = self.page
        data = self.data
        sfreq = self.sfreq
        interval = self.interval
        # plt.clf()
        for i in range(self.gridsize):
            if page*gridsize+i>=len(self.artefacts):
                ax  = self.axs[i]
                ax.clear()
                continue
            plotdata = data[(page*gridsize+i)*interval*sfreq:
                            (page*gridsize+i+1)*interval*sfreq]
            ax  = self.axs[i]
            ax.clear()
            ax.set_facecolor((1,1,1,1))   
            rr, yy = self.get_rrs(i, plotdata)
            ax.plot(plotdata, linewidth=0.5)
            ax.scatter(rr, yy , marker='x', color='r', linewidth=0.5,alpha=0.7)
            ax.text(0,ax.get_ylim()[1]+50,'{:.1f}%'.format(
                    self.kubios_art[page*gridsize+i]),fontsize=8)            
            xmin, middle, xmax = self._get_xlims(ax)
            ax.axvline(middle, color='gray', linewidth=0.65)     
            if self.artefacts[page*gridsize+i][0]:
                ax.axvspan(xmin, middle, facecolor=self.c_art, zorder=-100)
            if self.artefacts[page*gridsize+i][1]:
                ax.axvspan(middle, xmax, facecolor=self.c_art, zorder=-100)
            ax.set_xlim([xmin, xmax])
        self.draw()
            
        title = '{}\n{}/{}'.format(os.path.basename(self.file), 
                                  page, self.max_page)
        print('loading batch {}'.format(title))

        plt.suptitle(title)
        self.draw()


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
        
        
        helpstr = 'right\tnext page\n'\
                  'left\tprevious page\n'\
                  'enter\tjump to page X\n'\
                  'escape\tsave progress and quit\n'\
                  '\nmouse button\tmark as artefact\n\n'\
        
        if event.key=='escape':
            self.save()
            plt.close('all')
            return
        elif event.key =='enter':
            page = misc.input_box('Please select new page position', dtype=int, 
                                 initialvalue=self.page, minvalue=0, 
                                 maxvalue=self.max_page)
            if page:
                print('jumping to {}'.format(page))
                self.page = page
        elif event.key in ('right'):
            self.page += 1
        elif event.key=='left':
                self.page -= 1
        else:
            print(helpstr)
            print('unknown key {}'.format(event.key))
        if self.page<0:
            self.page=self.max_page
        elif self.page>self.max_page:
            self.page=0
        self.update()
        
        
    def toggle_artefact(self, part, idx):
        if part=='both':
            self.toggle_artefact('left', idx)
            self.toggle_artefact('right', idx)
            return
        if part=='left': i = 0
        if part=='right':  i = 1
        ax = self.axs[idx]
        xmin, middle, xmax = self._get_xlims(ax)
        art = self.artefacts[idx+self.page*self.gridsize][i]
        if art:
            ax.axvspan(middle if i else xmin, xmax if i else middle, 
                       facecolor=self.c_okay, zorder=-100)
            self.artefacts[idx+self.page*self.gridsize][i] = False
        else:
            ax.axvspan(middle if i else xmin, xmax if i else middle, 
                       facecolor=self.c_art, zorder=-100)
            self.artefacts[idx+self.page*self.gridsize][i] = True
        ax.set_xlim(xmin,xmax)
        self.save()

        
    def mouse_toggle_select(self, event):
        if event.inaxes is None:
            'Please click inside a plot'
            return
        idx = self.axs.index(event.inaxes)
        ax = self.axs[idx]          
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
            print('unknown button', event.button)
        plt.pause(0.001)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Load the visualizer for artefacts')
    parser.add_argument('-edf', '--edf_file', type=str,
                         help='A link to an edf-file. The channel ECG I  needs to be present.')
    parser.add_argument('-mat', '--mat_file', type=str,
                         help='A link to an mat-file created by Kubios.'
                              'It contains the RRs and the artefact annotation')
    parser.add_argument('-nrows', type=int, default=4,
                         help='Number of rows to display in the viewer')
    parser.add_argument('-ncols', type=int, default=4,
                         help='Number of columns to display in the viewer')
    parser.add_argument('-page', type=int, default=0,
                         help='At which page (epoch*gridsize) to start the viewer')
    args = parser.parse_args()
    edf_file = args.edf_file
    mat_file = args.mat_file
    nrows = args.nrows
    ncols = args.ncols
    page = args.page

    if edf_file is None:
        edf_file = misc.choose_file(exts=['edf', 'npy'], 
                                    title='Choose a EDF to display')
    print('loading {}'.format(edf_file))
    
    
    self = ECGPlotter(edf_file=edf_file, mat_file=mat_file, page=page,
                      nrows=nrows, ncols=ncols)
    plt.show(block=True)

