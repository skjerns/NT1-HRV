# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:09:10 2020

A small file to compare different QRS detectors

@author: Simon
"""

from pyedflib import highlevel
from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from viewer import viewer
from functions import resample
from external import processing
from ecgdetectors import Detectors



def compare()