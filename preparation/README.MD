## Data Preparation

### Summary

The following operations need to be done in order to work with the data:

1. Move EEG files and anonymize & rename
2. EEG channel names need to be renamed

#### 1. Move & Anonymize & Rename
In order to remove patient information from the file name, files need to be renamed. Anonymizing, moving and renaming can be done in one step.

Use the script `1_anonymize_rename.py`

Files will be stored at the 'data' location of the `config.py`

The new filenames will be calculated as following:

```
1. take lower case original filename and create adler32 hash with zlib.adler32()
2a. in case the hash contains a literal, convert to ASCII int and insert as string (but it shouldnt)
2. put resulting number in numpy seed, draw random float
3. multiply float with 1e8 and convert to int
4. new filename will be resulting int[:3] + '_' + int[3:] + .edf (_ for readability)

hash = zlib.adler32(filename.lower())
np.random.seed(hash)
rnd = np.random.rand()
number = int(rnd*1e8)
filename = str(number)[:3] + '_' + str(number)[3:] + '.edf'
```


#### 2. Rename channels
Channel names vary depending on the recording system.
For easier analysis we rename the channels to a common system and remove some channels that are not necessary for us (e.g. `battery`).

Use the script `2_rename_channels.py`

The files will be stored in the data folder/renamed. For safety reasons it is up to you to move them upwards to their correct position.