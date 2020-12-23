# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:46:37 2019

@author: Simon
"""
import os
import ospath
from unisens import utils
import pandas as pd
import numpy as np
from tkinter import  Tk
from unisens.utils import read_csv, write_csv
from tkinter.filedialog import askopenfilename, askdirectory
from collections import OrderedDict
from tkinter import simpledialog
from joblib import Memory
from pyedflib import highlevel
import hashlib
import warnings
import inspect
import datetime
import json

_cache = {}

def extract_ecg(edf_file, copy_folder):
    filename = os.path.basename(edf_file)
    new_edf_file = os.path.join(copy_folder, filename)
    if os.path.exists(new_edf_file): return
    try:
        header = highlevel.read_edf_header(edf_file)
    except:
        print(f'error in file {edf_file}')
        return
    channels = header['channels']
    try:
        channels.remove('cs_ECG')
    except:
        print(f'warning, {edf_file} has no cs_ECG')
    ch_names = [x for x in channels if 'ECG' in x.upper()]
    if len(ch_names)>1:
        print(f'Warning, these are present: {ch_names}, selecting {ch_names[0]}')
    ch_name = ch_names[0]

    signals, shead, header = highlevel.read_edf(edf_file, ch_names=[ch_name], digital=True, verbose=False)

    shead[0]['label'] = 'ECG'


    assert len(signals)>0, 'signal empty'
    try:
        highlevel.write_edf(new_edf_file, signals, shead, header, digital=True)
    except:
        shead[0]['digital_min'] = signals.min()
        shead[0]['digital_max'] = signals.max()
        highlevel.write_edf(new_edf_file, signals, shead, header, digital=True)



def save_results(classification_report, ss=None, clf=None,
                 subfolder=None , **kwargs):
    """
    Saves the current invocing script with results and metainformation

    The data will be dumped to a JSON file, with the datetime as name

    :param classification_report: output from sklearn.classification_report
    """
    report = classification_report # rename for shorter lines

    import config
    folder = os.path.join(config.documents, 'results')
    if subfolder is not None:
        folder = os.path.join(folder, subfolder)

    # add information about the sleepset that was used
    if ss:
        patients = ', '.join([p.code for p in ss])
        summary = ss.summary(verbose=False)
    else:
        patients = 'N/A'
        summary = 'N/A'

    # try to retrieve the current state of the sourcecode
    try:
        frame = inspect.stack()[1]
        main = inspect.getmodule(frame[0])
        code = inspect.getsource(main)
        file = frame.filename
        _cache['last_saved_file'] = file
    except:
        warnings.warn('Can\'t get source of Python console commands')
        file = _cache.get('last_saved_file', None)
        code = '## Code not found. trying to saving the current source ' + \
               '## state of last saved python file: {file}'
        if file:
            with open(file, 'r') as f:
                code = f.read()


    # create the filename.
    # The filename already contains the most important metric results
    avg = report["macro avg"]
    today = str(datetime.datetime.now()).rsplit('.',1)[0]

    filename =  f'f1 {avg["f1-score"]:.2f}, '
    filename += f'precision {avg["precision"]:.2f}, '
    filename += f'recall {avg["recall"]:.2f}, '
    filename += f'{str(clf)}; '
    filename += f"{today.replace(':', '.')}.json"


    obj = pd.Series({'clf': str(clf),
                     'date': today,
                     'report' : report,
                     'patients': patients,
                     'code': ''.join(code),
                     'summary': summary,
                     **kwargs})
    filename = os.path.join(folder, filename)
    obj.to_json(filename)

    # now also save in the summary
    summary_file = os.path.join(folder, '_summary.csv')
    if not os.path.exists(summary_file):
        line = 'Date, Class, Script, F1, Precision, Recall, '
        line += f"{', '.join(['True-' + str(x) for x in report['True'].keys()])}, "
        line += f"{', '.join(['False-' + str(x) for x in report['False'].keys()])}, "
        line += "File\n"
    else:
        line = ''
    with open(summary_file, 'a+') as f:

        line += f"{today}, {str(clf)}, {os.path.basename(file).replace(',', '_')}, "
        line += f"{avg['f1-score']}, {avg['precision']}, {avg['recall']}, "
        line += f"{', '.join([str(x) for x in report['True'].values()])}, "
        line += f"{', '.join([str(x) for x in report['False'].values()])}, "
        line += f"{os.path.basename(filename).replace(',', '_')} "
        f.write(line)

    return filename





def get_mnc_info():
    # first try to return cached value
    from xlrd import XLRDError
    try: return _cache['mnc_info']
    except: pass

    # if doesnt exist, load it
    import config
    file = os.path.join(config.folder_mnc, 'cohorts_deid.xlsx')

    try:
        df = pd.read_excel(file)
    except XLRDError:
        raise XLRDError('cant load xlsx, try installing pip install xlrd==1.2.0')

    mapping = {}
    for index, row in df.iterrows():
        ID = row.ID.upper().replace(' ', '_')
        if ID=='SUB001': ID = 'SUB01' # fix for difference in filename and info dict
        mapping[ID] = dict(row)

    _cache['mnc_info'] = mapping
    return mapping.copy()


def get_mapping():
    import config
    """gets the mapping dictionary for codes and names"""
    csv = os.path.join(config.documents, 'mapping_all.csv')
    mappings = utils.read_csv(csv)
    mappings.extend([x[::-1] for x in mappings]) # also backwards
    return dict(mappings)

def get_matching():
    import config 
    csv = os.path.join(config.documents, 'matching.csv')
    matching = utils.read_csv(csv, convert_nums=True)
    matching = [[code1, code2] for nt1,code1,_,_,cnt,code2,_,_,diff in matching if diff<99]
    matching.extend([x[::-1] for x in matching]) # also backwards
    return dict(matching)


def get_attribs():
    """get the attributes of the patients etc"""
    import config

    mappings = get_mapping()
    matching = get_matching()
    pre_coding_discard = [line[0] for line in read_csv(config.edfs_discard) if line[2]=='1']
    
    
    control = utils.read_csv(config.controls_csv)
    nt1 = utils.read_csv(config.patients_csv)
    
    control = [[c[0], {'gender':c[1].lower(), 'age':int(c[2]),'group':'control', 'drug_hrv':c[3], 'drug_sleep':c[4]}] for c in control]
    nt1 = [[c[0], {'gender':c[1].lower(), 'age':int(c[2]),'group':'nt1', 'drug_hrv':c[3], 'drug_sleep':c[4]}] for c in nt1]
    
    
    
    all_subjects = control + nt1
    all_subjects = dict([[mappings[x[0]],x[1]] for x in all_subjects if x[0] not in pre_coding_discard]) # convert to codified version
    
    # reverse mapping as well
    for c in all_subjects: all_subjects[c].update({'match':matching.get(c,'')})
    return dict(all_subjects)



def codify(filename): 
    """
    given a filename, will create a equal distributed
    hashed file number back to de-identify this filename
    """
    filename = filename.lower()
    m = hashlib.md5()
    m.update(filename.encode('utf-8'))
    hashing = m.hexdigest()
    hashing = int(''.join([str(ord(c)) for c in hashing]))
    hashing = hashing%(2**32-1) # max seed number for numpy
    np.random.seed(hashing)
    rnd = '{:.8f}'.format(np.random.rand())[2:]
    string = str(rnd)[:3] + '_' +  str(rnd)[3:]
    return string

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf = buf.reshape([h, w, 3])
    
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2 )
    return buf



def choose_file(default_dir=None, exts='txt', title='Choose file'):
    """
    Open a file chooser dialoge with tkinter.
    
    :param default_dir: Where to open the dir, if set to None, will start at wdir
    :param exts: A string or list of strings with extensions etc: 'txt' or ['txt','csv']
    :returns: the chosen file
    """
    root = Tk()
    root.iconify()
    root.update()
    if isinstance(exts, str): exts = [exts]
    name = askopenfilename(initialdir=None,
                           parent=root,
                           title = title,
                           filetypes =(*[("File", "*.{}".format(ext)) for ext in exts],
                                       ("All Files","*.*")))
    root.update()
    root.destroy()
    if not os.path.exists(name):
        print("No file chosen")
    else:
        return name

def choose_folder(default_dir=None, exts='txt', title='Choose file'):
    """
    Open a file chooser dialoge with tkinter.
    
    :param default_dir: Where to open the dir, if set to None, will start at wdir
    :param exts: A string or list of strings with extensions etc: 'txt' or ['txt','csv']
    :returns: the chosen file
    """
    root = Tk()
    root.iconify()
    root.update()
    if isinstance(exts, str): exts = [exts]
    name = askdirectory(initialdir=None,
                           parent=root,
                           title = title)
    root.update()
    root.destroy()
    if not os.path.exists(name):
        print("No folder chosen")
    else:
        return name
    
    
def input_box(message='Please type your input', title='Input', dtype=str,
              initialvalue = None, **kwargs):
    """
    Opens an input box.
    
    :param message: The message that is displayed at the prompt
    :param title: The title of the dialoge
    :param dtype: The dtype that is expected: [int, str, float]
    :param initialvalue: The default value in the dialoge
    :param kwargs: kwargs of the SimpleDialog class (eg minvalue, maxvalue)

    :returns: the given input of the user
    """    
    root = Tk()
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen
    root.update()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (ws/2)- w
    y = (hs/2)- h
    root.overrideredirect(1)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.withdraw()
    root.update()

    if dtype==str:
        dialoge = simpledialog.askstring
    elif dtype==int:
        dialoge = simpledialog.askinteger
    elif dtype==float:
        dialoge = simpledialog.askfloat
    else:
        raise ValueError('Unknown dtype: {}'.format(dtype))
        
    value = dialoge(title, message, initialvalue=initialvalue, parent=root,
                    **kwargs)     
    root.update()
    root.destroy()
    return value


class AttrDict(OrderedDict):
    """
    A dictionary that is ordered and can be accessed 
    by both dict[elem] and by dict.elem
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        
        
    # def __setattr__(self, item, value):
    #     return self.__setitem__( item, value)
    # def __getitem__(self, item, value):
    #     return self.__item__(item, value)
    
    # def __setitem__(self, item, value):
    #     print(4444444444)
    #     allowed = ''.join([str(chr(char)) for char in range(97,123)])
    #     allowed += ''.join([str(chr(char)).upper() for char in range(97,123)])
    #     allowed += ''.join([str(x) for x in  range(10)])
    #     reserved = ['False','def','if','raise','None','del','import','return',
    #                 'True','elif','in','try','and','else','is','while','as',
    #                 'except','lambda','with','assert','finally','nonlocal',
    #                 'yield','break','for','not','','class','from','or','',
    #                 'continue','global','pass']
    #     item = str(item)
    #     for char in item:
    #         if char not in allowed:
    #             item = item.replace(char, '_')
    #     if item in reserved:
    #         item += '_'
    #     print(item, value)
    #     OrderedDict.__setitem__(self, item, value)
    
    
