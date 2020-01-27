## Data Preparation

### Summary

The following operations need to be done in order to work with the data:

1. Move EEG files and anonymize & rename
2. EEG channel names need to be renamed

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


#### 2. Create matchings

We want to create a matching for our patients. The file `2_match_patients.py` will help you do so. It requires two files in the `documents` folder, `subjects_control.csv` and `subjects_nt1.csv`. Each line contains the original file name, the gender and the age at recording time.

The resulting list of matchings will be stored at `dropbox/nt1-hrv-documents/matching.csv`

#### 3. Separate NT1-set1 and NT1-Set2

As NT1-set1 has no scoring, we need to separate all matches with age difference maximum 5 years into different folders.

The script `3_copy_set1_set2.py` will do this. It will create folder 'set1' and 'set2' under `%cfg.data%/setX` and copy each patient file and it's corresponding control to this folder. Remark: no additional files (previous scoring, kubios files, artefact annotations) are copied






