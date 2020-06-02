# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:57:00 2020

RR_windows contains RR intervals in milliseconds!

@author: Simon Kern
"""
import logging as log
import config as cfg
import numpy as np
import scipy
from datetime import datetime
from scipy import stats
import hrvanalysis
from joblib.memory import Memory

### caching dir to prevent recomputation of reduntant functions
if hasattr(cfg, 'folder_cache'):
    print(f'caching enabeld in features.py to {cfg.folder_cache}')
    # memory = Memory(cfg.folder_cache, verbose=0)
    memory = Memory(None, verbose=0)
else:
    print('caching disabled in features.py')
    memory = Memory(None, verbose=0)
###################################################

    
def dummy(RR_windows, **kwargs):
    """
    each function here should be named exactly as the feature 
    name in config.features_mapping.
    Like this the feature extraction can be done automatically.
    The function can accept the following parameters:
        
    RR_windows: a list of windows with RR intervals denoted in SECONDS 
    
    IMPORTANT: Note that RRs are in SECONDS not MILLISECONDS
               and need to be converted if necessary inside the feature funcs
    
    all functions should accept **kwargs that are ignored.
    """
    pass
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)

    # Basic statistics
    mean_nni = np.mean(nn_intervals)
    median_nni = np.median(nn_intervals)
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int

    # Feature found on github and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni

    
    
# 1  
def mean_HR(RR_windows):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            HR = np.nan
        else:
            HR = len(wRR)/np.sum(wRR)*60
        feat.append(HR)
    return np.array(feat)

# 2
def mean_RR(RR_windows):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<1:
            mRR = np.nan
        else:
            mRR = np.mean(wRR)
        feat.append(mRR)
    return np.array(feat)


# 4
def RMSSD(RR_windows):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            RNSSD = np.nan
        else:
            diffRR = np.diff(wRR)
            RNSSD = np.sqrt(np.mean(diffRR ** 2))
        feat.append(RNSSD)
    return np.array(feat)

# 6
def pNN50(RR_windows):
    return pNNXX(RR_windows, XX=50)

def SDNN(RR_windows):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<1:
            SDNN = np.nan
        else:
            SDNN = np.std(wRR)
        feat.append(SDNN)
    return np.array(feat)

def SDSD(RR_windows):
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            SDSD = np.nan
        else:
            diffRR = np.diff(wRR)
            SDSD = np.std(diffRR)
        feat.append(SDSD)
    return np.array(feat)


# 10
def LF(RR_windows):
    feat = get_frequency_domain_features(RR_windows)['lf']
    return np.array(feat)
# 11
def HF(RR_windows):
    feat = get_frequency_domain_features(RR_windows)['hf']
    return np.array(feat)

# 12
def LF_HF(RR_windows):
    feat = get_frequency_domain_features(RR_windows)['lf_hf_ratio']
    return np.array(feat)
   
def pNNXX(RR_windows, XX=50):
    """
    Calculate the pNN index for a given millisecond interval difference
    pNN50 is the percentage of successive beats that differ more than 50ms
    """
    feat = []
    for wRR in RR_windows:
        if len(wRR)<2:
            pNN50 = np.nan
        else:
            diffRR = np.diff(wRR)
            pNN50 = ((diffRR>(XX/1000)).sum()/len(diffRR))*100
        feat.append(pNN50)
    return np.array(feat)

@memory.cache
def get_frequency_domain_features(RR_windows):
    """
    Calculate frequency domain features of this RR.
    This function is being cached as it computes a bunch of features
    at the same time 
    
    returns
    {    'lf': 0.0,
         'hf': 0.0,
         'lf_hf_ratio': nan,
         'lfnu': nan,
         'hfnu': nan,
         'total_power': 0.0,
         'vlf': 0.0}
    """ 
    assert isinstance(RR_windows, (list, np.ndarray))
    if isinstance(RR_windows, np.ndarray): 
        assert RR_windows.ndim==2, 'Must be 2D'
        
    if any([np.any(wRR>1000) for wRR in RR_windows if len(wRR)>0]):
        log.warn('Values seem to be in ms instead of seconds! Algorithm migh fail.')
        
    feats = { x:[] for x in ['lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf']}
    for wRR in RR_windows:
        if len(wRR)<2:
            for key, val in feats.items(): feats[key].append(np.nan)
        else:
            mRR = hrvanalysis.get_frequency_domain_features(wRR*1000)
            for key, val in mRR.items(): feats[key].append(val)
    return feats

#%% other functions

def _window_view(a, window, step = None, axis = None, readonly = True):
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

        step : tuple, int or None
            The offset between consecutive windows in desired dimension
            If None, offset is one in all dimensions
            If int, the offset for all windows over `axis`
            If tuple, the step along each `axis`.  
                `len(step)` must me equal to `len(axis)`

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
        if step:
            step = np.array(step, ndmin = 1)
            assert np.all(step > 0), "Only positive step allowed"
            assert (step.size == axs.size) | (step.size == 1), "step and axes don't match"
            stp[axs] = step

        astr = np.array(a.strides)

        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)

        as_strided = np.lib.stride_tricks.as_strided
        a_view = np.squeeze(as_strided(a, 
                                     shape = shape, 
                                     strides = strides, writeable=not readonly))
        
        return a_view


def extract_windows(signal, sfreq, wsize, step=None, pad=True):
    """ 
    Extract windows from a signal of a given window size with striding step
    
    :param wsize:  the size of the window
    :param stride: stepize of the window extraction. If None, stride=wsize
    :param pad:    whether to pad the array such that there are exactly 
                   len(signal)//stride windows (e.g. same as hypnogram)
    """ 
    assert signal.ndim==1
    if step is None: step = wsize
    n_step = len(signal)//step
    step *= sfreq
    wsize *= sfreq
    assert len(signal)>=wsize
    if pad: 
        padding = (wsize//2-step//2)
        signal = np.pad(signal, [padding, padding], mode='reflect')
    windows = _window_view(signal, window=wsize, step=step, readonly=True)
    if pad: 
        assert n_step == len(windows), 'unequal sizes'
    return windows

def extract_RR_windows(T_RR, RR, wsize, step=None, pad=True, 
                       expected_nwin=None):
    """ 
    Extract windows from a list of RR intervals of a given window size 
    with striding step. The windows are centered around the step borders.
    E.g. step=30, the first window will be centered around second 15,
    iff padding is activated.
    
    :param T_RR: the peak locations
    :param RR: a list of differences between RR peaks, e.g. [1.4, 1.5, 1.4]
    :param wsize:  the size of the window
    :param step: stepize of the window extraction. If None, stride=wsize
    :param pad:    whether to pad the array such that there are exactly 
                   len(signal)//stride windows (e.g. same as hypnogram)
    """ 
    
    # last detected peak should give roughly the recording length.
    # however, this is not always true, ie with a flat line at the end
    if T_RR[0]>1000: 
        raise ValueError(f'First peak at second {T_RR[0]}, seems wrong. Did you substract seconds after midnight?')
    record_len = int(T_RR[-1])
    if expected_nwin is None:
        expected_nwin = record_len//30 
    
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

    if step==None: step = wsize

    # pad left and right by reflecting the array to have
    # the same number of windows as e.g. hypnogram annotations
    if pad: 
        # this is how many seconds we need to add at the beginning and end
        pad_len_sec = (wsize//2-step//2)
        if pad_len_sec>0:
            # these are the positions of this second in the RR array
            pad_rr_i_l = second_idxs[pad_len_sec]
            pad_rr_i_r = second_idxs[len(second_idxs)-pad_len_sec]
            
            # These are the values that we want to add before and after
            pad_rr_l = RR[:pad_rr_i_l]
            pad_rr_r = RR[pad_rr_i_r:]
            RR = np.hstack([pad_rr_l, RR, pad_rr_r])
            
            # we also need to re-adapt the second_idxs, to know at which second
            # which RR is now.
            pad_sec_l = second_idxs[:pad_len_sec]
            pad_sec_r = second_idxs[-pad_len_sec:] 
            pad_sec_r = pad_sec_r + pad_sec_r[-1] -  pad_sec_r[0] + pad_sec_l[-1]+2
            second_idxs = second_idxs + pad_sec_l[-1] + 1
            second_idxs = np.hstack([pad_sec_l, second_idxs, pad_sec_r])
        
    # assert second_idxs[-1]==len(RR)-1
    # these are the centers of the windows, exactly between two step boundaries
    windows = []
    n_windows = int(len(second_idxs)//step)-wsize//step+1
    for i in range(n_windows):
        # get RR values for this window
        if i*step>=len(second_idxs) or i*step+wsize>=len(second_idxs):
            windows.append(np.array([]))
            continue
        idx_start = second_idxs[i*step]
        idx_end = second_idxs[i*step+wsize]
        wRR = RR[idx_start:idx_end]
        windows.append(wRR)
    # assert expected_nwin==len(windows)
    return windows
   
   
def artefact_detection(T_RR, RR, wsize=30, step=None):
    """
    Scans RR interval arrays for artefacts.
    
    Returns an array with wsize windows and the number of 
    artefacts in this window. 
    
    good examples: 
        flat line with no flat line: 106_06263, 107_13396
        flat line once in a while: 659_88640
        todo: look for tiny ecg as well
        
    The following artefact correction procedure is applied:
        2. If any RR is > 2 (==HR 30, implausible), discard
        3. If n_RRi == 0: Epoch ok.
        4. If n_RRi<=2: Epoch ok. (keep RRi)
        5. If n_RRi==3: 
        If n_RRi are not consecutive: Epoch ok (keep RRi)
        else: discard
        6. If n_RRi>=4: Discard epoch.
        
    """
    if step is None: step = wsize

    idxs = extract_RR_windows(T_RR, np.arange(len(RR)), wsize, step=step)

    # RR_pre is before correction
    # RR_post is after correction (as coming directly from Kubios)
    RR_pre = np.diff(T_RR)
    RR_post = RR
    windows_RR_pre =  [RR_pre[idx] if len(idx)>0 else [] for idx in idxs]
    windows_RR_post = [RR_post[idx] if len(idx)>0 else [] for idx in idxs]
    assert len(windows_RR_pre)==len(windows_RR_post)
    
    art = []
    for w_RR_pre, w_RRi_post in zip(windows_RR_pre, windows_RR_post):
        
        assert len(w_RR_pre) == len(w_RRi_post)
        # 2. HR<30, seems odd + too large or too small
        if len(w_RR_pre)<15 or np.argmax(w_RR_pre>2) or np.argmax(w_RRi_post<0.4):
            art.append(True)
            continue
        else:
            diff = np.where(not np.isclose(w_RR_pre, w_RRi_post))[0]
            # 5. special case, are they consecutive or not?
            if len(diff)==3:
                # are the consecutive? Then the summary of their distance should be 3
                if np.sum(diff)==3:
                    art.append(True)
                    continue
            if len(diff)>3:
                art.append(True)
                continue
        # if we reach this far, no artefact has been detected
        art.append(False)
        
    art = np.array(art)
    assert len(art)==len(windows_RR_post)
    return art

    
#%% main 
if __name__=='__main__':
    
    
    ## testing flatline detection
    from sleep import Patient
    p = Patient('Z:/NT1-HRV-unisens/659_60515')
    kubios = p.feats.get_data()['Data']
    RR = kubios['RR']
    T_RR = kubios['T_RR']-p.startsec
    wsize = 300
    step = None
    artefact_detection(RR,T_RR, wsize, step)
    RR_windows = extract_RR_windows(T_RR, RR, wsize, step=30)
    ## testing RR window extraction

    # from sleep import Patient
    # p = Patient('Z:/NT1-HRV-unisens/009_08813')
    # data = p.feats.get_data()['Data']
    # RR = data['RR']
    
    # T_RR = data['T_RR']
    # start = datetime.strptime(p.timestampStart,'%Y-%m-%dT%H:%M:%S')
    # T_RR -= (start.second + start.minute*60 + start.hour*3600)
    # expected_nwin = p.epochs_hypno
    # wsize = 30
    # step = 30
    # pad = False
