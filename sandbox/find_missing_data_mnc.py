# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:02:53 2020

@author: Simon
"""
import config as cfg
import misc
import ospath

# %% print info
if __name__=='__main__':
    data_folder = cfg.folder_mnc
    files = ospath.list_files(data_folder, exts=['edf'], subfolders=True)

    info = misc.get_mnc_info()
    fullfiles = files.copy()
    files = [f[:-9] for f in files] # remove extension & "-nsrr"
    files = [ospath.basename(file).replace(' ', '_') for file in files]

    nt1 = []
    hyp = []
    cnt = []
    missing_file = []
    missing_info = []
    missing_hypno = []
    missing_info_missing_hypno = []
    for name, full in zip(files, fullfiles):
        folder = ospath.dirname(full)
        has_hypno = True if len(ospath.list_files(folder, patterns=f'*{name}*.xml')) else False
        if name.upper() in info:

            item = info[name.upper()].copy()
            diagnosis = item['Diagnosis']
            if has_hypno:
                if 'CONTROL' in diagnosis:
                    cnt.append(item)
                elif 'T1' in diagnosis:
                    nt1.append(item)
                elif 'OTHER HYPERSOMNIA' in diagnosis:
                    hyp.append(item)

            else:
                missing_hypno.append(full)

            del info[name.upper()]

        else:
            if has_hypno:
                missing_info.append(full)
            else:
                missing_info_missing_hypno.append(full)


    for name, value in info.items():
        missing_file.append(f'{info[name]["Cohort"].lower()}/{info[name]["ID"]}')



    # Parallel(n_jobs=10)(delayed(to_unisens)(
            # edf_file, unisens_folder=unisens_folder, skip_exist=False, overwrite=False) for edf_file in tqdm(files, desc='Converting'))

    # single process
    # for file in tqdm(files):
        # to_unisens(file, unisens_folder=unisens_folder, skip_exist=False, overwrite=False)
