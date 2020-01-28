# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:46:37 2019

@author: Simon
"""
import os
from tkinter import  Tk
from tkinter.filedialog import askopenfilename
from collections import OrderedDict
from tkinter import simpledialog



def write_csv(csv_file, data_list, sep=';'):
    """
    Parameters
    ----------
    csv_file : str
        a filename.
    data_list : list
        a list of list. each list is a new line, each list of list is an entry there.
    sep : str, optional
        the separator to be used. The default is ';'.

    Returns
    -------
    lines : TYPE
        DESCRIPTION.

    """
    with open(csv_file, 'w') as f:
        string = '\n'.join([';'.join(line) for line in data_list])
        f.write(string)
    return True

def read_csv(csv_file, sep=';'):
    """
    simply load an csv file with a separator and newline as \\n
    comments are annotated as starting with # and are removed
    empty lines are removed
    
    :param csv_file: a csv file to load
    :param sep: set a different separator. this is language specific
    """
    with open(csv_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        lines = [line for line in lines if not line.startswith('#')]
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line!='']
        lines = [line.split(';') for line in lines]
    return lines

def choose_file(default_dir=None,exts='txt', title='Choose file'):
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
        print("No file exists")
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