# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:46:37 2019

@author: Simon
"""
import os
from unisens import utils
import numpy as np
from tkinter import  Tk
from unisens.utils import read_csv, write_csv
from tkinter.filedialog import askopenfilename, askdirectory
from collections import OrderedDict
from tkinter import simpledialog
import hashlib


def set_time_xticks(ax, sfreq=256):
    """
    given an axis will set the xticks of the axis to time annotations
    """
    pass

def get_mapping():
    import config
    """gets the mapping dictionary for codes and names"""
    csv = os.path.join(config.documents, 'mapping_all.csv')
    mappings = utils.read_csv(csv)
    mappings.extend([x[::-1] for x in mappings]) # also backwards
    return dict(mappings)
    
    
def get_attribs():
    import config
    """get the attributes of the patients etc"""
    mappings = get_mapping()
    control_csv = os.path.join(config.documents, 'subjects_control.csv')
    nt1_csv = os.path.join(config.documents, 'subjects_nt1.csv')
    
    control = utils.read_csv(control_csv)
    nt1 = utils.read_csv(nt1_csv)

    control = [[c[0], {'gender':c[1].lower(), 'age':int(c[2]),'group':'control'}] for c in control] 
    nt1 = [[c[0], {'gender':c[1].lower(), 'age':int(c[2]),'group':'nt1'}] for c in nt1] 
    
    all = control + nt1
    all = [[mappings[x[0]],x[1]] for x in all ]
    return dict(all)

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
    
    
class CaseInsensitiveDict(OrderedDict):
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, str) else key

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()
    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self.__class__._k(key))
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self.__class__._k(key), value)
    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(self.__class__._k(key))
    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self.__class__._k(key))
    def has_key(self, key):
        return super(CaseInsensitiveDict, self).has_key(self.__class__._k(key))
    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(self.__class__._k(key), *args, **kwargs)
    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(self.__class__._k(key), *args, **kwargs)
    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(self.__class__._k(key), *args, **kwargs)
    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))
    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)