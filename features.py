# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:57:00 2020

@author: nd269
"""
import config as cfg
import numpy as np
import scipy
from scipy import stats

# create empty dummy functions
_locals = locals()
for key, name in cfg.mapping_feats.items():
    _locals[name] = lambda *args, **kwargs:  False
    
    
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