### Introduction and Usage

This viewer is meant as an easy tool to annotate ECG artefact epochs
with the help of Kubios Premium, an HRV analysis tool.
The viewer needs two files:

1. An EDF with a channel names "ECG I"
2. A .mat file created by Kubios that contains the detected RR peaks. This .mat file needs to be called %EDFFILE%_hrv.mat

You can start the viewer with the command `python artefact_viewer` or  on Windows
withthe batch file `start_viewer.bat`.
The viewer will then automatically ask you which file you want to choose
and will automatically infer the .mat filename, else ask you to provide one.

Otherwise the viewer can also be started with parameters

`python artefact_viewer.py -PARAMETERS`

optional arguments:
```
  -h, --help            show this help message and exit
  -edf EDF_FILE, --edf_file EDF_FILE
                        A link to an edf-file. The channel ECG I needs to be
                        present. The .mat file will be tried to matched automatically
  -mat MAT_FILE, --mat_file MAT_FILE
                        A link to an mat-file created by Kubios.It contains
                        the RRs and the artefact annotation
  -nrows NROWS          Number of rows to display in the viewer
  -ncols NCOLS          Number of columns to display in the viewer
  -pos POS              At which position (epoch) to start the viewer
```
  
  
### Artefact annotation guidelines

The goal is to annotate the ECG files in blocks of 15 second based on whether 
the an artefact is present and all RR peaks are set correctly.
The RR peaks are detected by Kubios, but are sometimes placed incorrectly.
They are marked with a red cross. If the cross is not at the peak
of the signal, it is not correct.
Kubios also has an automatic artefact detection mechanism.
That means there are already artefacts annotated. However,
the algorithm sometimes makes mistakes that need to be corrected
(this is your task).

Definition:
An epoch is marked if
- A flat line is present anywhere in the signal
- There is too much noise such that RR peaks are not correct
- RR peaks seem odd or off (too much distance between RRs)

Generally: Don't be too strict, if there is any doubt whether there is an 
		   artefact or not, annotate an artefact. Your task is to be loose,
		   another persons job will be to double check and be strict.


### Viewer controls

The viewer is easy and intuitive to use. To make analysis faster,
there are several signals loaded at once (standard batch of 16 epochs).
This number can also be set with command line parameters (`ncols`, `nrows`)
if you prefer to see more or less signals at once.

The following buttons can be pressed:

	Left mouse:  Mark whole epoch as artefact/no artefact
	Right mouse: Mark half epoch as artefact/no artefact
	Enter/right: load next batch of signals
	Left:        load previous batch of signals
	Escape:      close the viewer
	
The results of the annotation will be saved to `%EDFFILE%.npy`.
This file will be updated and loaded automatically so you in case
the viewer crashes, nothing is lost.

