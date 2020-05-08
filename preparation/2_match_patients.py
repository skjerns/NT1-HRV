# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:16:26 2020

Script that creates age and gender mapping based on 
csv files for nt1 and controls

@author: skjerns
"""
import ospath
import config as cfg
import misc

def read_subjects(file):
    with open(file, 'r', encoding='utf8') as f:
      entries = f.read().strip()

    entries = entries.split('\n')
    entries = [entry for entry in entries if not entry.startswith('#')]
    entries = [p.split(';') for p in entries]
    entry_dict = dict({p[0].strip():{'age':p[2],'gender':p[1].strip()} for p in entries})
    return entry_dict
    

def read_csv(file):
    with open(file, 'r', encoding='utf8') as f:
      entries = f.read().strip()
    entries = entries.replace('"', '').split('\n')
    entries = [p.split(';') for p in entries]
    entry_dict = dict({p[0].strip():p[1].strip() for p in entries})
    return entry_dict

#%% main
if __name__ == '__main__':      
    # get the mappings from names to codes
    mappings = misc.get_mapping()
    
    # get the list of all subjects and controls
    patients_all = read_subjects(cfg.patients_csv)
    controls_all = read_subjects(cfg.controls_csv)
    
    # ignore these items when creating the matching
    to_discard = [line[0] for line in misc.read_csv(cfg.edfs_discard) if line[2]=='1']

    controls = controls_all.copy()
    patients = patients_all.copy()
    
    
    matches = []
    
    # iteratively go through all age differences and try to find matches
    # within that range. This way we should get an somewhat optimal matching
    for i in range(cfg.max_age_diff+1):
        matches_i = []
        for p_name, attrs in patients.copy().items():
            
            # if this file is absolutely broken, discard it (no eeg and no ecg).
            if p_name in to_discard: continue
            # retrieve attributes
            p_gender = attrs['gender'].strip().lower()
            p_age = int(attrs['age'])
            p_code = mappings[p_name]
            
            # loop through all controls and check if we find a match
            for c_name, attrs in controls.copy().items():
                # if this file is absolutely broken, discard it (no eeg and no ecg).
                if c_name in to_discard: continue
                # retrieve attributes
                c_gender = attrs['gender'].strip().lower()
                c_age = int(attrs['age'].strip())
                c_code = mappings[c_name]
                
                # gender doesnt match? skip
                if c_gender!=p_gender: continue 
            
                # if age diff is within the wanted age range i we have a match
                age_diff = abs(p_age-c_age)
                if age_diff > i: continue
                matches_i.append(f'{p_name}; {p_code}; {p_gender}; {p_age}; {c_name}; {c_code}; {c_gender}; {c_age}; {age_diff}')
                # now delete both items from the list of patients and controls
                # so that we dont match them again and stop the loop
                controls.pop(c_name)
                patients.pop(p_name)
                break
        matches.append(matches_i)
    
            
    # now we loop over all patients that did not get any match
    # 'patients' should have all of those left in it
    not_matched = []
    for p_name, attrs in patients.copy().items():
        # if this file is absolutely broken, discard it (no eeg and no ecg).
        if p_name in to_discard: continue
        p_gender = attrs['gender'].strip().lower()
        p_age = int(attrs['age'])
        not_matched.append(f'{p_name}; {p_code}; {p_gender}; {p_age} ; ; ; ; ;99')
        
    
    # now we create the csv_string that we will write to a file:
    lines = ['#Patient Name; Patient Code; Patient Gender; Patient Age; Control Name; Control Code; Control Gender; Control Age; Difference']
    for diff, match_i in enumerate(matches): #last one
        lines += [''] # add empty line before each new age diff section
        lines += [f'# +-{diff} age difference, {len(match_i)} matchings']
        lines.extend(match_i)
    
    lines += ['']
    lines += [f'# No match for {len(not_matched)} patients']
    lines += not_matched

    # now we add all controls that are in the project
    lines += ['']
    lines += ['# Already used controls']
    for c_name in controls_all:
        lines += [f'{c_name};;;;;;;;99']
        
    matching_csv = ospath.join(cfg.documents, 'matching.csv')
    misc.write_csv(matching_csv, lines)
        
        
        
        
        
        
        
        
        
        
        
        