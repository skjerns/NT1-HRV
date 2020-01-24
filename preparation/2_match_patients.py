# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:16:26 2020

Script that creates age and gender mapping based on 
csv files for nt1 and controls

@author: skjerns
"""
import ospath
import config as cfg

def read_subjects(file):
    with open(file, 'r') as f:
      entries = f.read().strip()
    entries = entries.split('\n')
    entries = [p.split(';') for p in entries]
    entry_dict = dict({p[0].strip():{'age':p[2],'gender':p[1].strip()} for p in entries})
    return entry_dict
    
def read_csv(file):
    with open(file, 'r') as f:
      entries = f.read().strip()
    entries = entries.replace('"', '').split('\n')
    entries = [p.split(';') for p in entries]
    entry_dict = dict({p[0].strip():p[1].strip() for p in entries})
    return entry_dict

if __name__ == '__main__':  
    documents = cfg.documents
    datasets = [ospath.join(documents, 'mapping_' + d + '.csv') for d in cfg.datasets]
    
    mappings = {}
    for dataset in datasets: 
        mappings.update(read_csv(dataset))
    
    patients_csv = ospath.join(documents, 'subjects_nt1.csv')
    controls_csv = ospath.join(documents, 'subjects_control.csv')

    patients = read_subjects(patients_csv)
    controls = read_subjects(controls_csv)
    
    
    csv_string = ''
    max_age_diff = 100
    
    matches = []
    for i in range(max_age_diff):
        match = {}
        for p_name, attr in list(patients.items()):
            p_age    = attr['age']
            p_gender = attr['gender'].lower()
            for c_name, attr in controls.items():
                c_age    = attr['age']
                c_gender = attr['gender'].lower()
                if c_gender!=p_gender:
                    continue
                diff = abs(int(p_age)-int(c_age))
                if diff>i:
                    continue
                print('{} matches to {} with {} diff'.format(p_name, c_name, diff))
                match[p_name] = c_name
                del patients[p_name]
                del controls[c_name]
                break
        matches.append(match)
        if len(match)!=0:
            csv_string += '\n+-{} age difference, {} matchings\n'.format(i, len(match))
            for patient, control in match.items():
                patient_mapping = mappings.get(patient, 'NOTFOUND')
                control_mapping = mappings.get(control, 'NOTFOUND')
                csv_string += '{};{};{};{}\n'.format(patient, patient_mapping, control, control_mapping)
    matching_csv = ospath.join(documents, 'matching.csv')
    with open(matching_csv, 'w') as f:
        f.write(csv_string)
        
        
        
        
        
        
        
        
        
        
        
        