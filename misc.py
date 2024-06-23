# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:46:37 2019

@author: Simon
"""
import os
import ospath
import hashlib
import warnings
import inspect
import datetime
import json
import seaborn as sns
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
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt



_cache = {}
try:
    stack = inspect.stack()[1:]
    for frame in stack:
        if '.py' in frame.filename:
            _cache['last_saved_file'] = frame.filename
            break
except:
    pass


def low_priority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)




def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         value_format.format(h), ha="center",
                         va="center")

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

def get_auc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return auc(fpr, tpr)


def save_roc(filename, y_true, y_prob, title_add=''):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_prob)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC, area under curve: {roc_auc[1]:.3f}\n'+title_add)
    plt.legend(['ROC curve'],loc="lower right")
    plt.savefig(filename)

def save_dist(filename, y_true, y_prob, title_add=''):
    plt.figure()
    nt1 = y_prob[y_true.astype(bool)]
    cnt = y_prob[~y_true.astype(bool)]
    sns.histplot(nt1, bins=25, kde=True, label='NT1')
    sns.histplot(cnt, bins=25, kde=True, label='Controls')
    plt.xlabel('Predicted probability for NT1')
    plt.legend()
    plt.title('Distribution of probabilities\n'+title_add)
    plt.savefig(filename)


def save_results(y_true, y_prob, name, params=None, ss=None, clf=None,
                 subfolder=None , **kwargs):
    """
    Saves the current invocing script with results and metainformation

    The data will be dumped to a JSON file, with the datetime as name

    :param classification_report: output from sklearn.classification_report
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = y_prob.argmax(1)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    import config

    base_folder = os.path.join(config.documents, 'results')
    if subfolder is not None:
        folder = os.path.join(base_folder, subfolder)
    else:
        folder = base_folder

    # add information about the sleepset that was used
    if ss:
        patients = ', '.join([p.code for p in ss])
        summary = str(ss.summary(verbose=False))
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
        file = _cache.get('last_saved_file', 'unknown')
        code = '## Code not found. trying to saving the current source ' + \
               '## state of last saved python file: {file}'
        if file and file != 'unknown':
            with open(file, 'r') as f:
                code = f.read()


    # create the filename.
    # The filename already contains the most important metric results
    avg = report["macro avg"]
    today = str(datetime.datetime.now()).rsplit('.',1)[0]

    filename = f"{today.replace(':', '.')} - {name}.json"

    obj = pd.Series({'name':name,
                     'clf': str(clf),
                     'date': today,
                     'y_true': y_true,
                     'y_pred': y_prob,
                     'report' : report,
                     'patients': patients,
                     'code': ''.join(code),
                     'summary': summary,
                     **kwargs})

    # input('Done. Press <enter> to save results.\n')
    obj.to_json(os.path.join(folder, filename))


    save_roc(os.path.join(folder, filename[:-4] + '_roc.png'), y_true, y_prob[:,1],
             title_add = name)
    save_dist(os.path.join(folder, filename[:-4] + '_dist.png'), y_true, y_prob[:,1],
              title_add = name)

    # now also save in the summary
    summary_file = os.path.join(base_folder, '_summary.csv')
    # check if we need to write a header file
    if not os.path.exists(summary_file):
        line = 'sep=,\n' # tell Excel what we are using
        line += 'Date, Name, Params, Script, F1, Precision, Recall, '
        line += f"{', '.join(['True-' + str(x) for x in report['True'].keys()])}, "
        line += f"{', '.join(['False-' + str(x) for x in report['False'].keys()])}, "
        line += "JSON-file\n"
    else:
        line = ''

    with open(summary_file, 'a+') as f:
        jsonname = filename.replace(',', '_')
        scriptname = os.path.basename(file).replace(',', '_')
        line += f"{today}, {name}, {scriptname}, "
        line += f"{avg['f1-score']}, {avg['precision']}, {avg['recall']}, "
        line += f"{', '.join([str(x) for x in report['True'].values()])}, "
        line += f"{', '.join([str(x) for x in report['False'].values()])}, "
        line += f"{jsonname} \n"
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
        ID = str(row.ID).upper().replace(' ', '_')
        if ID=='SUB001': ID = 'SUB01' # fix for difference in filename and info dict
        mapping[ID] = dict(row)

    _cache['mnc_info'] = mapping
    return mapping.copy()


def get_mapping():
    import config
    """gets the mapping dictionary for codes and names"""
    csv = os.path.join(config.documents, 'mapping_all.csv')
    # csv_mnc = os.path.join(config.documents, 'mapping_mnc.csv')

    mappings = utils.read_csv(csv)
    # mappings_mnc = utils.read_csv(csv_mnc)
    # mappings.extend(mappings_mnc)
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


def make_fig(
    n_axs=30,
    bottom_plots=2,
    no_ticks=False,
    suptitle="",
    xlabel="Lag in ms",
    ylabel="Sequenceness",
    figsize=None,
    despine=True,
):
    """
    helper function to create a grid space with RxC rows and a
    large row with two axis on the bottom

    returns: fig, axs(size=(rows*columns)), ax_left_bottom, ax_right_bottom
    """

    COL_MULT = 10  # to accomodate also too large axis
    # some heuristic for finding optimal rows and columns
    for columns in [2, 4, 6, 8]:
        rows = np.ceil(n_axs / columns).astype(int)
        if columns >= rows:
            break
    assert columns * rows >= n_axs

    if isinstance(bottom_plots, int):
        bottom_plots = [1 for _ in range(bottom_plots)]
    n_bottom = len(bottom_plots)
    COL_MULT = 1
    if n_bottom > 0:
        for COL_MULT in range(1, 12):
            if (columns * COL_MULT) % n_bottom == 0:
                break
        if not (columns * COL_MULT) % n_bottom == 0:
            warnings.warn(
                f"{columns} cols cannot be evenly divided by {bottom_plots} bottom plots"
            )
    fig = plt.figure(dpi=75, constrained_layout=True, figsize=figsize)
    # assuming maximum 30 participants
    gs = fig.add_gridspec(
        (rows + 2 * (n_bottom > 0)), columns * COL_MULT
    )  # two more for larger summary plots
    axs = []

    # first the individual plot axis for each participant
    for x in range(rows):
        for y in range(columns):
            ax = fig.add_subplot(gs[x, y * COL_MULT : (y + 1) * COL_MULT])
            if no_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            axs.append(ax)

    fig.suptitle(suptitle)

    if len(bottom_plots) == 0:
        return fig, axs

    # second the two graphs with all data combined/meaned
    axs_bottom = []
    step = np.ceil(columns * COL_MULT // n_bottom).astype(int)
    for b, i in enumerate(range(0, columns * COL_MULT, step)):
        if bottom_plots[b] == 0:
            continue  # do not draw* this plot
        ax_bottom = fig.add_subplot(gs[rows:, i : (i + step)])
        if xlabel:
            ax_bottom.set_xlabel(xlabel)
        if ylabel:
            ax_bottom.set_ylabel(ylabel)
        if i > 0 and no_ticks:  # remove yticks on righter plots
            ax_bottom.set_yticks([])
        axs_bottom.append(ax_bottom)
    if despine:
        sns.despine(fig)
    return fig, axs, *axs_bottom


def normalize_lims(axs, which='both'):
    """for all axes in axs: set function to min/max of all axs


    Parameters
    ----------
    axs : list
        list of axes to normalize.
    which : string, optional
        Which axis to normalize. Can be 'x', 'y', 'xy' oder 'both'.

    """
    if which=='both':
        which='xy'
    for w in which:
        ylims = [getattr(ax, f'get_{w}lim')() for ax in axs]
        ymin = min([x[0] for x in ylims])
        ymax = max([x[1] for x in ylims])
        for ax in axs:
            getattr(ax, f'set_{w}lim')([ymin, ymax])

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
