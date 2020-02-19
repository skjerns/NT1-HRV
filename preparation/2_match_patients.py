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
    with open(file, 'r', encoding='utf8') as f:
      entries = f.read().strip()
    entries = entries.split('\n')
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


if __name__ == '__main__':  
    documents = cfg.documents
    datasets = [ospath.join(documents, 'mapping_' + d + '.csv') for d in cfg.datasets]
    
    mappings = {}
    for dataset in datasets: 
        mappings.update(read_csv(dataset))
    
    patients_csv = ospath.join(documents, 'subjects_nt1.csv')
    controls_csv = ospath.join(documents, 'subjects_control.csv')

    patients_all = read_subjects(patients_csv)
    controls_all = read_subjects(controls_csv)

    controls = controls_all.copy()
    patients = patients_all.copy()
    
    csv_string = '#Patient Name; Patient Code; Patient Gender; Patient Age; Control Name; Control Code; Control Gender; Control Age; Difference'
    max_age_diff = cfg.max_age_diff
    
    matches = []
    used_controls = ['A5941', 'A3558', 'B0007', 'B0025', 'A9308']
    for i in range(max_age_diff+1):
        match = {}
        for p_name, attr in list(patients.items()):
            p_age    = attr['age'].strip()
            p_gender = attr['gender'].lower()
            for c_name, attr in controls.items():
                c_age    = attr['age'].strip()
                c_gender = attr['gender'].lower()
                if c_gender!=p_gender:
                    continue
                diff = abs(int(p_age)-int(c_age))
                if diff>i:
                    continue
                print('{} matches to {} with {} diff'.format(p_name, c_name, diff))
                match[p_name] = c_name
                used_controls.append(c_name)
                del patients[p_name]
                del controls[c_name]
                break
        matches.append(match)
        if len(match)!=0:
            csv_string += '\n#\n#+-{} age difference, {} matchings\n'.format(i, len(match))
            for patient, control in match.items():
                patient_mapping = mappings[patient]
                control_mapping = mappings[control]
                p_gender = patients_all[patient]['gender']
                p_age    = patients_all[patient]['age']
                c_gender = controls_all[control]['gender']
                c_age    = controls_all[control]['age']
                csv_string += f'{patient};{patient_mapping}; {p_gender}; {p_age}; {control}; {control_mapping}; {c_gender}; {c_age};{i}\n'
   
    csv_string += '\n#No matching\n'
    for p_name, attr in list(patients.items()):
        p_age    = attr['age']
        p_gender = attr['gender'].lower()
        p_code   = mappings[patient]
        csv_string+=f'{p_name}; {p_code}; {p_gender}; {p_age};;;;;99\n'
        
    csv_string += '\n#Already used controls\n'
    for c in used_controls:
        csv_string += f'{c};;;;;;;;99\n'
    matching_csv = ospath.join(documents, 'matching.csv')
    
    with open(matching_csv, 'w') as f:
        f.write(csv_string)
        
        
        
        
        
        
        
        
        
        
        
        