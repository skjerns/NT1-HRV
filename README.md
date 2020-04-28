# NT1-HRV
Experimental biomarker research for early detection of Narcolepsy type 1

## work in progress... ##

## setup

to install the packages needed for this repository, run the following command:
```
pip install tqdm pyedflib h5py mat73 dateparser joblib pytablewriter
pip install git+https://github.com/skjerns/skjerns-utils
```

Additionally, add the local repository folder to your `$PYTHONPATH` user-path variable.

## configuration

there are two ways to store user or machine specific information:

Privacy-insensitive settings and variables can go into `config.py` and can be loaded with. See the code template there and add a `elif`-statement with your machine name or user name
```Python
import config as cfg
value = cfg.setting_name
```

Additionally a configuration file for privacy-sensitive information exists (ie. file names, folder names, etc). It is located in the shared dropbox folder `ntv-hrv-documents/user_variables.py`. This script will **not** be uploaded to GitHub. This script will automatically be imported when importing setting, that means all variables added to `user_variables.py` will be accessible via `config.py`
```Python
import config as cfg
value = cfg.private_user_variable
```






