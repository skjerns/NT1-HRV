# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:57:00 2020

@author: nd269
"""
import config as cfg
import numpy as np
import scipy
from datetime import datetime
from scipy import stats

# # create empty dummy functions
# _locals = locals()
# for key, name in cfg.mapping_feats.items():
#     _locals[name] = lambda *args, **kwargs:  False
   
    



def dummy(ecg, **kwargs):
    """
    each function here should be named exactly as the feature 
    name in config.features_mapping.
    Like this the feature extraction can be done automatically.
    The function can accept the following parameters:
        
    ecg, rr, kubios
    
    all functions should accept **kwargs that are ignored.
    """
    
def HR(RR):
    pass
    
# 1 

def mean_HR(kubios, **kwargs):
    data = kubios['TimeVar']['mean_HR']
    return data.squeeze()

# 2
def mean_RR(kubios, **kwargs):
    data = kubios['TimeVar']['mean_RR']
    return data.squeeze()

# 4
def RMSSD(kubios, **kwargs):
    data = kubios['TimeVar']['RMSSD']
    return data.squeeze()

# 6
def pNN50(kubios, **kwargs):
    data = kubios['TimeVar']['pNNxx']
    return data.squeeze()

# 10
def LF(kubios, **kwargs):
    data = kubios['TimeVar']['LF_power']
    return data.squeeze()

# 11
def HF(kubios, **kwargs):
    data = kubios['TimeVar']['HF_power']
    return data.squeeze()

# 12
def LF_HF(kubios, **kwargs):
    data = kubios['TimeVar']['LF_power']/kubios['TimeVar']['HF_power']
    return data.squeeze()



def _window_view(a, window, steps = None, axis = None, readonly = True):
        """
        Create a windowed view over `n`-dimensional input that uses an 
        `m`-dimensional window, with `m <= n`

        Parameters
        -------------
        a : Array-like
            The array to create the view on

        window : tuple or int
            If int, the size of the window in `axis`, or in all dimensions if 
            `axis == None`

            If tuple, the shape of the desired window.  `window.size` must be:
                equal to `len(axis)` if `axis != None`, else 
                equal to `len(a.shape)`, or 
                1

        steps : tuple, int or None
            The offset between consecutive windows in desired dimension
            If None, offset is one in all dimensions
            If int, the offset for all windows over `axis`
            If tuple, the steps along each `axis`.  
                `len(steps)` must me equal to `len(axis)`

        axis : tuple, int or None
            The axes over which to apply the window
            If None, apply over all dimensions
            if tuple or int, the dimensions over which to apply the window

        generator : boolean
            Creates a generator over the windows 
            If False, it will be an array with 
                `a.nidim + 1 <= a_view.ndim <= a.ndim *2`.  
            If True, generates one window per .next() call
        
        readonly: return array as readonly

        Returns
        -------

        a_view : ndarray
            A windowed view on the input array `a`, or a generator over the windows   

        """
        ashp = np.array(a.shape)
        if axis != None:
            axs = np.array(axis, ndmin = 1)
            assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
        else:
            axs = np.arange(ashp.size)

        window = np.array(window, ndmin = 1)
        assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
        wshp = ashp.copy()
        wshp[axs] = window
        assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

        stp = np.ones_like(ashp)
        if steps:
            steps = np.array(steps, ndmin = 1)
            assert np.all(steps > 0), "Only positive steps allowed"
            assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
            stp[axs] = steps

        astr = np.array(a.strides)

        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)

        as_strided = np.lib.stride_tricks.as_strided
        a_view = np.squeeze(as_strided(a, 
                                     shape = shape, 
                                     strides = strides, writeable=not readonly))
        
        return a_view


def extract_windows(signal, sfreq, wsize, steps=None, pad=True):
    """ 
    Extract windows from a signal of a given window size with striding steps
    
    :param wsize:  the size of the window
    :param stride: stepsize of the window extraction. If None, stride=wsize
    :param pad:    whether to pad the array such that there are exactly 
                   len(signal)//stride windows (e.g. same as hypnogram)
    """ 
    assert signal.ndim==1
    if steps is None: steps = wsize
    n_steps = len(signal)//steps
    steps *= sfreq
    wsize *= sfreq
    assert len(signal)>=wsize
    if pad: 
        padding = (wsize//2-steps//2)
        signal = np.pad(signal, [padding, padding], mode='reflect')
    windows = _window_view(signal, window=wsize, steps=steps, readonly=True)
    if pad: 
        assert n_steps == len(windows), 'unequal sizes'
    return windows


def extract_RR_windows(T_RR, RR, wsize, steps=None, pad=True):
    """ 
    Extract windows from a list of RR intervals of a given window size 
    with striding steps. The windows are centered around the step borders.
    E.g. step=30, the first window will be centered around second 15,
    iff padding is activated.
    
    :param T_RR: the peak locations
    :param RR: a list of differences between RR peaks, e.g. [1.4, 1.5, 1.4]
    :param wsize:  the size of the window
    :param stride: stepsize of the window extraction. If None, stride=wsize
    :param pad:    whether to pad the array such that there are exactly 
                   len(signal)//stride windows (e.g. same as hypnogram)
    """ 
    windows = []
    
    # last detected peak should give roughly the recording length.
    # however, this is not always true, ie with a flat line at the end
    record_len = int(T_RR[-1])
    
    # this array gives us the position of the RR at second x
    # e.g. seconds_idxs[5] will return the RR indices starting at second 5.
    second_idxs = []
    c = 0 
    for i in range(record_len):
        while i>=T_RR[c]:
            c+=1
        second_idxs.append(c)
    second_idxs = np.array(second_idxs)
    
    
    assert record_len==len(second_idxs), f'record len={record_len}, but seconds array is {len(second_idxs)}'

    
    # pad left and right by reflecting the array to have
    # the same number of windows as e.g. hypnogram annotations
    # if pad: 
    #     # this is how much we need
    #     pad_len = (wsize//2-steps//2)
    #     # take first pad_len RR values and reflect them back.
    #     pad_l = RR[:np.argmax(seconds>pad_len)][::-1]
    #     pad_r = RR[-np.argmax(seconds>pad_len):][::-1]
    #     RR = np.hstack([pad_l, RR, pad_r])
    #     cumsum = np.cumsum(RR)
    #     # divide by stepsize to know for each index to which 
    #     # window it belongs
    #     windows = cumsum//1
    
    # these are the centers of the windows, exactly between two step boundaries
    n_windows = int(record_len//steps)-wsize//steps+1
    for i in range(n_windows):
        # get RR values for this window
        idx_start = second_idxs[i*steps]
        idx_end = second_idxs[i*steps+wsize]
        wRR = RR[idx_start:idx_end]
        windows.append(wRR)
        
        
    
if __name__=='__main__':
    from sleep import Patient
    p = Patient('Z:/NT1-HRV-unisens/009_08813')
    data = p.feats.get_data()['Data']
    RR = data['RR']
    T_RR = data['T_RR']
    start = datetime.strptime(p.timestampStart,'%Y-%m-%dT%H:%M:%S')
    T_RR -= (start.second + start.minute*60 + start.hour*3600)
    wsize = 300
    steps = 30
