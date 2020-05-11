## Data Preparation

### Summary

The following operations need to be done in order to work with the data:

1. Move EEG files and anonymize & rename
2. Create age and gender matching for patients and controls
3. Convert files to unisens for better data structure


#### 1. Move & Anonymize & Rename
In order to remove patient information from the file name, files need to be renamed. Anonymizing, moving,  renaming and channel streamlining can be done in one step.

Use the script `1_anonymize_rename.py`

Files will be stored at the 'data' location of the `config.py`

The new filenames will be calculated as following:

```
1. take lower case original filename and create an md5 hash
2. convert all chars to their ASCII ordinal and concatenate
2. make modulo 2**32-1 (max int) and put resulting number in numpy seed, draw random float
3. print first 8 digits of the float
4. new filename will be resulting float[:3] + '_' + float[3:] + .edf (_ for readability)

hash = md5(filename.lower())
hash_int = int(join''.([str(ord(i)) for i in hash)])%(2**32-1)
np.random.seed(hash)
rnd = np.random.rand()
number = '{:.8f}'.format(rnd)[2:]
filename = str(number)[:3] + '_' + str(number)[3:] + '.edf'
```

The patient name and the birthdate will be stripped from the header.

Channel names vary depending on the recording system.
For easier analysis we rename the channels to a common system. The ECG channel will always be named 'ECG I'

The folder locations are declared in `Dropbox\nt1-hrv-documents\user_variables.py \datasets` and the target folder is declared at `Dropbox\nt1-hrv-documents\user_variables.py \folder_edf`


#### 2. Create matchings

We want to create a matching for our patients. The file `2_match_patients.py` will help you do so. It requires two files in the `documents` folder, `subjects_control.csv` and `subjects_nt1.csv`. Each line contains the original file name, the gender and the age at recording time. The maximum age difference for matching will be stored in `config.cfg` 

The resulting list of matchings will be stored at `dropbox/nt1-hrv-documents/matching.csv`


#### 3. Create Unisens data structure


Next step comprises extracting relevant information from the EDF and saving it as Unisens. `sleep.py` contains a class `Patient` that extends `Unisens`. Here we save features, hypnograms etc without having a monolith-edf-file. 

ECG, EEG, EMG and EOG will be extracted as well as the features from Kubios, EDF annotations and hypnograms and additional information.

For this, the `MAT`-files that were created by Kubios need to be added to the data folder, next to the EDF file. Additionally the hypnograms need to be added. Currently we don't have an automatic way to extract these information.

The script `4_convert_to_unisens.py` will convert all EDFs in the data folder to the unisens folder, that can be set in the `user_variables.py`. All edfs at `folder_edf` will be converted to unisens at `folder_unisens` 





### Deprecated


#### (3. Separate NT1-set1 and NT1-Set2)

(this step is deprecated as we don't differentiate between set1 and set2 anymore. This file has been moved to /sandbox)

As NT1-set1 has no scoring, we need to separate all matches with age difference maximum 5 years into different folders.

The script `3_copy_set1_set2.py` will do this. It will create folder 'set1' and 'set2' under `%cfg.data%/setX/*` and copy each patient file and it's corresponding control to this folder. Remark: no additional files (previous scoring, kubios files, artefact annotations) are copied.

Additionally it will copy all non-matched files into `%cfg.data%/setX/not_matched/*`
