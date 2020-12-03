# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:40:41 2020

automate kubios

@author: Simon
"""

import os
import autoit
import time
from tqdm import tqdm


edf_folder = ''
kubios_title = 'Kubios HRV Standard 3.4.2'
title_open = 'Open Data File'




def extract_rr(filename):
    pid = autoit.run("C:/Program Files/Kubios/Kubios HRV Standard/kubioshrv.exe")
    
    # wait until screen is loaded
    assert autoit.win_wait_active(f"[TITLE:{kubios_title}]", 30)
    time.sleep(0.25)
    
    # get position and click menu->open
    x, y, _, _ = autoit.win_get_pos(f"[TITLE:{kubios_title}]")
    assert autoit.mouse_click(x=x+20, y=y+50, speed=10)
    time.sleep(0.5)
    assert autoit.mouse_click(x=x+20, y=y+70, speed=10)
    
    # wait for loading of file input screen
    assert autoit.win_wait_active(f"[TITLE:{title_open}]", 30)
    time.sleep(0.25)
    assert autoit.control_send(f"[TITLE:{title_open}]", "[CLASS:Edit]", filename)
    time.sleep(0.25)
    assert autoit.control_click(f"[TITLE:{title_open}]", "[CLASS:Button; INSTANCE:1]")
    time.sleep(1)
    
    # check if there was an error (if Open Data File is still there)
    assert not autoit.win_active(f"[TITLE:{title_open}]"), 'Failed to load!'
    
    assert os.system(f"taskkill /PID {pid} /F")==0


edf_files = [x for x in os.listdir(edf_folder) if x.endswith('.edf')]

for filename in tqdm(edf_files):
    extract_rr(filename)