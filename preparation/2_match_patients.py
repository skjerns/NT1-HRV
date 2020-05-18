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
import pandas as pd
from pymatch.Matcher import Matcher
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

def read_subjects(file):
    with open(file, 'r', encoding='utf8') as f:
      entries = f.read().strip()
    entries = entries.split('\n')
    entries = [entry for entry in entries if not entry.startswith('#')]
    entries = [p.split(';') for p in entries]
    entry_dict = dict({p[0].strip():{'age':int(p[2]),'gender':p[1].strip().lower()} for p in entries})
    return entry_dict
    

def read_csv(file):
    with open(file, 'r', encoding='utf8') as f:
      entries = f.read().strip()
    entries = entries.replace('"', '').split('\n')
    entries = [p.split(';') for p in entries]
    entry_dict = dict({p[0].strip():p[1].strip() for p in entries})
    return entry_dict


def create_list_possible_matches(patients, controls, agediff=3):
    possible = {p:[] for p in patients}
    for p, p_age in patients.items():
        for c, c_age in controls.items():
            if abs(p_age-c_age)<=agediff:
                possible[p].append(c)
    possible = {p:possible[p] for p in possible if not len(possible[p])==0}
    return possible


def exhaustive_matching(patients, controls, agediff=3, possible=None):
    """obviously NP-hard, so not computable, but I tried for fun!"""
    if possible is None:
        possible = create_list_possible_matches(patients, controls, agediff)
        
    all_matches = []
    for patient in possible:
        for possible_match in possible[patient]:
            matches = {patient:possible_match}
            possible_sub = possible.copy()
            possible_sub[patient] = possible_sub[patient].copy()
            possible_sub[patient].remove(possible_match)
            sub_matches = exhaustive_matching(patients, controls, agediff=agediff, 
                                              possible=possible_sub)
            matches.update(sub_matches)
            all_matches.append(matches)
    if len(all_matches)>0: all_matches = max(all_matches, key=len)
    return all_matches
    

def check_matches_unique(matches, not_matched):
    all_matches = []
    for match_i in matches:
        all_matches.extend(match_i)
    patients, controls = zip(*all_matches)
    
    assert len(set(controls))==len(all_matches), f'only {len(set(controls))} unique controls but {len(all_matches)} matches'
    assert len(set(patients))==len(all_matches), f'only {len(set(patients))} unique patients but {len(all_matches)} matches'
    assert len(set(not_matched))==len(not_matched), f'only {len(set(not_matched))} unique patients but {len(not_matched)} matches'
    for not_match in not_matched: assert not_match not in patients, f'{not_match} is not matched but also in patients'
    for not_match in not_matched: assert not_match not in controls, f'{not_match} is not matched but also in controls'
    
    
def pymatch_matching(patients, controls, max_age_diff=3):
    raise Exception('This doesnt seem to work')
    patients = {p:patients[p] for p in patients}
    controls = {p:controls[p] for p in controls}
    
    p_names = list(patients)
    p_ages = [p['age'] for p in patients.values()]
    p_genders = [p['gender'] for p in patients.values()]
    p_group = [1 for _ in patients]
    patients_df = pd.DataFrame(list(zip(p_names,p_genders, p_ages, p_group)), columns=['Name','Gender', 'Age', 'Group'])

    c_names = list(controls)
    c_ages = [c['age'] for c in controls.values()]
    c_genders = [c['gender']for c in controls.values()]
    c_group = [0 for _ in controls]
    controls_df = pd.DataFrame(list(zip(c_names, c_genders, c_ages, c_group)), columns=['Name','Gender', 'Age', 'Group'])

    matches = [[] for _ in range(max_age_diff+1)]
    not_matched = []
    
    for gender in ['male', 'female']:
        test_group = patients_df.loc[patients_df['Gender']==gender]
        control_group = controls_df.loc[controls_df['Gender']==gender]
        m = Matcher(test_group , control_group , yvar='Group', exclude = ['Name', 'Gender'])
        m.fit_scores(balance=True, nmodels=100, formula ='')
        m.match(with_replacement=False, nmatches=1, threshold=10)
        for match in m.matched_data.loc[m.matched_data['Group']==0].itertuples():
            case = test_group.iloc[match.match_id]
            diff = abs(case.Age - match.Age)
            if diff<=max_age_diff:
                matches[diff].append([case.Name, match.Name])
            else:
                not_matched.append(case.Name)
                
    return matches, not_matched


def bootstrap_matchings(patients, controls, iterations=100000, max_age_diff=3):

    res = Parallel(n_jobs=8, prefer='processes')(delayed(random_matching)(*pc, seed=i, 
                max_age_diff=max_age_diff) for i, pc in tqdm(enumerate([[patients.copy(), 
                                                                         controls.copy()]]*iterations), 
                                       desc='Bootstrapping', total=iterations))
                                                                 
    best_matches = []
    best_not_matched = list(patients)
    best_diff = np.inf
    
    for matches, not_matched, mean_diff in res:
        if (len(best_not_matched)>len(not_matched)) or\
            (len(best_not_matched)==len(not_matched) and best_diff>mean_diff):
            best_not_matched = not_matched
            best_matches = matches
            best_diff = mean_diff
        if len(best_not_matched)==0:break
    return best_matches, best_not_matched

def random_matching(patients, controls, seed=0, max_age_diff=3):
    np.random.seed(seed)
    diffs = []
    matches = [[] for _ in range(max_age_diff+1)]
    not_matched = list(patients)
    for p_name, attrs in patients.items():
        p_gender = attrs['gender']
        p_age = int(attrs['age'])
        c_names = list(controls)
        idxs = np.arange(len(c_names))
        np.random.shuffle(idxs)
        for idx in idxs:
            c_name = c_names[idx]
            c_gender = controls[c_name]['gender']
            c_age = controls[c_name]['age']
            if c_gender!=p_gender:continue
            diff = abs(c_age-p_age)
            if diff>cfg.max_age_diff:continue
            controls.pop(c_name)
            diffs.append(diff)
            matches[diff].append([p_name, c_name])
            not_matched.remove(p_name)
            break
    return matches, not_matched, np.mean(diffs)

def greedy_matching(patients, controls, max_age_diff):
    matches = []
    patients = patients.copy()
    # iteratively go through all age differences and try to find matches
    # within that range. This way we should get an somewhat optimal matching
    for i in range(max_age_diff+1):
        matches_i = []
        for p_name, attrs in patients.copy().items():
            
            # if this file is absolutely broken, discard it (no eeg and no ecg).
            # retrieve attributes
            p_gender = attrs['gender']
            p_age = attrs['age']
            
            # loop through all controls and check if we find a match
            for c_name, attrs in controls.copy().items():
                # if this file is absolutely broken, discard it (no eeg and no ecg).
                # retrieve attributes
                c_gender = attrs['gender']
                c_age = attrs['age']
                
                # gender doesnt match? skip
                if c_gender!=p_gender: continue 
            
                # if age diff is within the wanted age range i we have a match
                age_diff = abs(p_age-c_age)
                if age_diff > i: continue
                matches_i.append([f'{p_name}', f'{c_name}'])
                # now delete both items from the list of patients and controls
                # so that we dont match them again and stop the loop
                controls.pop(c_name)
                patients.pop(p_name)
                break
        matches.append(matches_i)
    
            
    # now we loop over all patients that did not get any match
    # 'patients' should have all of those left in it
    not_matched = []
    for p_name, attrs in patients.items():
        # if this file is absolutely broken, discard it (no eeg and no ecg).
        p_gender = attrs['gender']
        p_age = attrs['age']
        not_matched.append(f'{p_name}')
    return matches, not_matched

#%% main
if __name__ == '__main__':      
    max_age_diff = cfg.max_age_diff
    
    # get the mappings from names to codes
    mappings = misc.get_mapping()

    # get the list of all subjects and controls
    patients_all = read_subjects(cfg.patients_csv)
    controls_all = read_subjects(cfg.controls_csv)
    
    # ignore these items when creating the matching
    to_discard = [line[0] for line in misc.read_csv(cfg.edfs_discard) if line[2]=='1']
    controls = {c:controls_all[c] for c in controls_all.copy() if not c in to_discard}
    patients = {p:patients_all[p] for p in patients_all.copy() if not p in to_discard}

    # matches, not_matched = greedy_matching(patients.copy(), controls.copy(), max_age_diff=max_age_diff)
    matches, not_matched = bootstrap_matchings(patients.copy(), controls.copy(), iterations=10000000, max_age_diff=max_age_diff)
    # matches, not_matched = pymatch_matching(patients.copy(), controls.copy())
    
    check_matches_unique(matches, not_matched)
    
    # now we create the csv_string that we will write to a file:
    lines = ['#Patient Name; Patient Code; Patient Gender; Patient Age; Control Name; Control Code; Control Gender; Control Age; Difference']
    for diff, match_i in enumerate(matches): #last one
        lines += [''] # add empty line before each new age diff section
        lines += [f'# +-{diff} age difference, {len(match_i)} matchings']
        for p_name, c_name in match_i:
            p_code = mappings[p_name]
            p_gender = patients_all[p_name]['gender']
            p_age = patients_all[p_name]['age']
            c_code = mappings[c_name]
            c_gender = controls_all[c_name]['gender']
            c_age = controls_all[c_name]['age']
            lines.append(f'{p_name}; {p_code}; {p_gender}; {p_age}; {c_name}; {c_code}; {c_gender}; {c_age}; {diff}')
    
    lines += ['']
    lines += [f'# No match for {len(not_matched)} patients']
    lines += [f'{m}; {patients_all[m]["gender"]}; {patients_all[m]["age"]};;;;;;99' for m in not_matched]

    # now we add all controls that are in the project
    lines += ['']
    lines += ['# Already used controls']
    for c_name in controls_all:
        lines += [f'{c_name}; {controls_all[c_name]["gender"]}; {controls_all[c_name]["age"]};;;;;;99']
        
    matching_csv = ospath.join(cfg.documents, 'matching.csv')
    misc.write_csv(matching_csv, lines)
        
