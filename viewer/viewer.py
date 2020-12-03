# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:28:49 2019

Generalized ECG viewer that can also plot dots/markers

@author: Simon
"""
import os
import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import logging
import datetime
from cycler import cycler
from pyedflib import highlevel

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.INFO)



class ECGPlotter():

    def _handle_close(self,evt):
        plt.close('all')
        return

    def __init__(self, data, fs, markers={}, interval=30, nrows=3, ncols=3,
                  startpage=0, title='', verbose=True):
        """
        :param data: ECG data or any other continuous data
        :param fs: sample rate of the signal
        :markers: a dictionary, where each item contains the name and the
                  occurence of the marker in seconds
        :param interval: 
        """
        if len(markers)>4:
            print('More than 4 markers currently not supported')

        self.flipped = False
        self.markers = markers
        self.page = startpage
        self.interval = interval
        self.nrows = nrows
        self.ncols = ncols
        self.gridsize = nrows*ncols
        self.title = title
        self.verbose = verbose

        # load 
        self.data = data
        self.fs = fs
        self.max_page = len(data)//fs//self.interval//self.gridsize

        # set up the plot, connect the button presses
        self.axs = []
        self.fig, axs = plt.subplots(nrows, ncols)
        self.axs = np.array(axs).flatten()

        _ = self.fig.canvas.mpl_connect("button_press_event", self.mouse_toggle_select)
        _ = self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        _ = self.fig.canvas.mpl_connect('close_event', self._handle_close)
        self.background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)

        self.colors = ['b', 'g', 'r', 'k']
        self.styles = ['x', '*', '+', 'o']
        self.update()



    def draw(self):
        self.fig.canvas.draw()

    def get_marker(self, plot_nr, marker_name, plotdata):
        """ get the marker height (yy) given for certain seconds"""
        marker_sec = self.markers.get(marker_name)
        plotstart_sec = (self.page*self.gridsize+plot_nr)*self.interval
        idx_start = np.searchsorted(marker_sec, plotstart_sec)
        idx_stop  = np.searchsorted(marker_sec, plotstart_sec+self.interval)
        marker_samples = marker_sec[idx_start:idx_stop]*self.fs
        yy = self.data[marker_samples.round().astype(int)]
        xx = marker_samples-plotstart_sec*self.fs
        return xx, yy


    #%% update
    def update(self):
        gridsize = self.gridsize
        page = self.page
        data = self.data
        fs = self.fs
        interval = self.interval
        markers = self.markers

        formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x//self.fs)}')

        for i in range(self.gridsize):

            if page*gridsize+i>=len(self.data):
                ax  = self.axs[i]
                ax.clear()
                continue
            plotdata = data[(page*gridsize+i)*interval*fs:
                            (page*gridsize+i+1)*interval*fs]
            ax  = self.axs[i]
            ax.clear()

            for j, marker_name in enumerate(markers):
                xx, yy = self.get_marker(i, marker_name, plotdata)
                scatter = 0 #markerpoints.max()*0.005*j
                m = self.styles[j]
                c = self.colors[j]
                ax.scatter(xx+scatter, yy, marker=m, color=c,
                           linewidth=1, alpha=0.65)

            if i==0:
                self.fig.legend(list(markers))
            ax.plot(plotdata, linewidth=0.5)
            # ax.set_xlim([0, self.interval*self.fs])
            ax.set_xlabel('seconds')
            ax.xaxis.set_major_formatter(formatter)
            seconds = (page*gridsize+i)*interval
            timestr1 = (datetime.timedelta(seconds=seconds))
            timestr2 = (datetime.timedelta(seconds=seconds+interval))
            ax.set_title(f'{timestr1} - {timestr2}', fontsize=11)


        title = f'ECGPlotter: {self.title}\n{page}/{self.max_page}'
        title += ' - flipped'*self.flipped
        plt.suptitle(title)
        if self.verbose:
            print('printing batch {}'.format(title))

        self.draw()
        # plt.tight_layout()


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

    #%% key press
    def key_press(self, event):


        helpstr = 'right\tnext page\n'\
                  'left\tprevious page\n'\
                  'enter\tjump to page X\n'\
                  'escape\tquit\n'\
                  '\nmouse button\tmark as artefact\n\n'\

        if event.key=='escape':
            plt.close('all')
            return

        elif event.key =='enter':
            page = misc.input_box('Please select new page position', dtype=int,
                                 initialvalue=self.page, minvalue=0,
                                 maxvalue=self.max_page)
            if page and self.verbose:
                print('jumping to {}'.format(page))
                self.page = page
        elif event.key in ('right'):
            self.page += 1
        elif event.key=='left':
                self.page -= 1
        elif event.key=='u':
            self.data = -self.data
            if self.verbose: print('flipping u/d')
            self.flipped = not self.flipped
        else:
            if self.verbose:
                print(helpstr)
                print('unknown key {}'.format(event.key))
        if self.page<0:
            self.page=self.max_page
        elif self.page>self.max_page:
            self.page=0
        self.update()


    def mouse_toggle_select(self, event):
        if event.inaxes is None:
            'Please click inside a plot'
            return
        idx = np.where(self.axs==event.inaxes)[0]
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
            if self.verbose:
                print('unknown button', event.button)
        plt.pause(0.001)

#%% main
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Load the visualizer for artefacts')
    parser.add_argument('-edf', '--edf_file', type=str,
                         help='A link to an edf-file. The channel ECG I  needs to be present.')
    parser.add_argument('-nrows', type=int, default=2,
                         help='Number of rows to display in the viewer')
    parser.add_argument('-ncols', type=int, default=2,
                         help='Number of columns to display in the viewer')
    parser.add_argument('-page', type=int, default=0,
                         help='At which page (epoch*gridsize) to start the viewer')
    args = parser.parse_args()
    edf_file = args.edf_file
    nrows = args.nrows
    ncols = args.ncols
    page = args.page

    if edf_file is None:
        edf_file = misc.choose_file(exts=['edf', 'npy'],
                                    title='Choose a EDF to display')
    print('loading {}'.format(edf_file))


    channels = highlevel.read_edf_header(edf_file)['channels']
    ch_nr = channels.index([ch for ch in channels if 'ECG' in ch.upper()][0])
    data, sheader, header = highlevel.read_edf(edf_file, ch_nrs=ch_nr)
    data = data[0]
    fs = sheader[0]['sample_rate']
    title = os.path.basename(edf_file)
    self = ECGPlotter(data=data, fs=fs, startpage=page,
                      nrows=nrows, ncols=ncols, title=title)
    plt.show(block=True)


