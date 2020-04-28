# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:01:56 2020

@author: skjerns
"""
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import config as cfg
from pytablewriter import TableWriterFactory, HtmlTableWriter
from pytablewriter.style import Style
import plotting
sns.set(style='whitegrid')
#%% settings


report_dir = os.path.join(cfg.documents, 'reports')
os.makedirs(report_dir, exist_ok=True)

table_format = 'html' # can be html, csv, xlsx, md ...

css_format = """
<style>
table {border-collapse: collapse;}
table {border: 1px solid black;}
th {background-color: #4CAF50;color: white;}
tr:nth-child(even) {background-color: #f2f2f2;}
table {width: 750px;}
img {max-width:750px;
  width: auto;
  height: auto;
}
</style>
"""

writer = TableWriterFactory().create_from_file_extension(table_format)
writer.max_workers = 1
writer.column_styles = [Style(align="left")] + [Style(align="center")]*10
writer.margin = 2

#%%


def lineplot_table(table, title, columns=3, rows=None, save_to=None, 
                   xlabel=None, ylabel=None):
    """plot a table as figure and save to png
    
    a table is defined as a dictionary with 3 or 4 levels and has the following
    items
    
    dictionary['variable name']['group1/group2']['values'] = [[0,5,2,3,4, ...],] NxX matrix
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]
    """
    n_plots = len(table)

    if rows is None:
        size = (int(np.ceil(n_plots/columns)), columns)
    if columns is None:
        size = (rows, int(np.ceil(n_plots/rows)))
    
    c = {'nt1':'b', 'control':'r'}                
    
    fig, axs = plt.subplots(*size)
    axs = axs.flatten()
    for i, descriptor in enumerate(table): 
        ax = axs[i]
        for group in ['nt1', 'control']:
            values_nt1 = table[descriptor][group]['values']
            x = np.arange(values_nt1.shape[-1]) + (0 if group=='nt1' else 0.035*values_nt1.shape[-1])
            # mean values of feature
            y_mean = np.nanmean(values_nt1, 0)
            # upper std
            sem = stats.sem(values_nt1, 0)
            # lower std
            # err_kws = {'x':x, 'y1':y1, 'y2':y2, 'alpha':0.2, 'color':c[group]}
            # sns.pointplot(x=x, y=y_mean, ax=ax, c=c[group])
            ax.errorbar(x, y_mean, yerr=sem, c=c[group], fmt='-o', alpha=0.7)

        
        # convert sleep stage to stage name if necessary
        if descriptor in [0,1,2,3,4,5]: 
            descriptor = cfg.num2stage[descriptor]
        if isinstance(descriptor, tuple):
            descriptor = '-'.join([str(cfg.num2stage[d]) for d in descriptor])
        if not isinstance(descriptor, str): 
            descriptor=str(descriptor)
        ax.set_title(descriptor)
        ax.legend(['NT1', 'Control'])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    plt.suptitle(title, y=1)
    plt.pause(0.01)
    plt.tight_layout() 
    
    if save_to is None:
        save_to = os.path.join(cfg.documents, 'reports', f'{title}.png' )
    fig.savefig(save_to)
    return fig, axs


def distplot_table(table, title, columns=3, rows=None, save_to=None):
    """plot distribution plot of a table and save to png
    
    a table is defined as a dictionary with 3 or 4 levels and has the following
    items
    
    dictionary['variable name']['group1/group2']['values'] = [0,5,2,3,4, ...]
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]
    """
    n_plots = len(table)

    if rows is None:
        size = (int(np.ceil(n_plots/columns)), columns)
    if columns is None:
        size = (rows, int(np.ceil(n_plots/rows)))
                    
    fig, axs = plt.subplots(*size)
    axs = axs.flatten()
    for i, descriptor in enumerate(table): 
        ax = axs[i]
        values_nt1 = table[descriptor]['nt1']['values']
        values_cnt = table[descriptor]['control']['values']
        bins = min(15, len(np.unique(values_nt1)), len(np.unique(values_cnt)))
        try: sns.distplot(values_nt1,bins=bins, ax=ax)
        except: pass
        try: sns.distplot(values_cnt,bins=bins, ax=ax)
        except: pass
        
        p_val = plotting.format_p_value(table[descriptor]['p'], bold=False)
        # convert sleep stage to stage name if necessary
        if descriptor in [0,1,2,3,4,5]: 
            descriptor = cfg.num2stage[descriptor]
        if isinstance(descriptor, tuple):
            descriptor = '-'.join([str(cfg.num2stage[d]) for d in descriptor])
        if not isinstance(descriptor, str): descriptor=str(descriptor)
        ax.set_title(descriptor + f' - p {p_val}')
        ax.legend(['NT1', 'Control'])
        
    plt.suptitle(title, y=1)
    plt.pause(0.01)
    plt.tight_layout() 
    
    if save_to is None:
        save_to = os.path.join(cfg.documents, 'reports', f'{title}.png' )
    fig.savefig(save_to)
    return fig, axs
        


def print_table(table, title):
    """
    Format a dictionary as a table and save it to HTML/MD/CSV
    
    example dictionary
    dictionary['variable name']['group1/group2']['values'] = [0,5,2,3,4, ...]
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]
 
    """
    report_dir = os.path.join(cfg.documents, 'reports')
    
    matrix = []    
    writer.table_name = title
    writer.headers = ["Variable", "NT1", "Control", "p"]
    
    for name, d in table.items():
        nt1_mean, nt1_std = d['nt1']['mean'], d['nt1']['std']
        c_mean, c_std = d['control']['mean'], d['control']['std']
        p = format_p_value(d['p'])
        matrix += [[name, f'{nt1_mean:.2f} ± {nt1_std:.2f}', f'{c_mean:.2f} ± {c_std:.2f}', p]]
        
    writer.value_matrix  = matrix
    file = os.path.join(report_dir, f'{title}.{table_format}')
    
    # writer = 
    string = writer.dumps()
    possible_plot = f'<br><br><br><br><img src="{title}.png" alt="not found: {title}.png">'
    with open(file, 'w') as f:
        f.write(css_format + string + possible_plot)
    return string

def print_table_with_subvars(table, title):
    """
    Format a dictionary as a table and save it to HTML/MD/CSV
    
    example dictionary
    dictionary['variable name']['group1/group2']['values'] = [0,5,2,3,4, ...]
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]
 
    """
    report_dir = os.path.join(cfg.documents, 'reports')

    matrix = []    
    writer.table_name = title
    writer.headers = ["Variable", "Subvar", "NT1", "Control", "p"]
    
    for name, subtable in table.items():
        matrix+= [[fbold(name)]]
        for subvar in subtable:

            nt1_mean, nt1_std = subtable[subvar]['nt1']['mean'], subtable[subvar]['nt1']['std']
            c_mean, c_std = subtable[subvar]['control']['mean'], subtable[subvar]['control']['std']
            p = format_p_value(subtable[subvar]['p'])
            if subvar in [0,1,2,3,4,5]:
                subvar = cfg.num2stage[subvar]
            matrix += [['',subvar, f'{nt1_mean:.2f} ± {nt1_std:.2f}', f'{c_mean:.2f} ± {c_std:.2f}', p]]
            
    writer.value_matrix  = matrix
    file = os.path.join(report_dir, f'{title}.{table_format}')
    
    # writer = 
    string = writer.dumps()
    possible_plot = f'<br><br><br><br><img src="{title}.png" alt="not found: {title}.png">'
    with open(file, 'w') as f:
        f.write(css_format + string + possible_plot)
    return string


def fbold(string):
    """turns a string bold in the given format"""
    if table_format=='html':
        return f'<b>{string}</b>'
    if table_format=='md':
        return f'**{string}**'
    return string

def fitalic(string):
    """turns a string bold in the given format"""
    if table_format=='html':
        return f'<i>{string}</i>'
    if table_format=='md':
        return f'*{string}*'
    return string


def format_p_value(p, bold=True):
    if  isinstance(p, str): return p
    if p==0:
        p='-'
    elif p>0.1:
        p = f'{p:.2f}'
    elif p>0.05:
        p = f'{p:.3f}'
    elif p>0.001:
        p = f'{p:.3f}*'
        p = fbold(p) if bold else p
    elif p>0.0001:
        p = '>0.001**'
        p = fbold(p) if bold else p
    elif p>0.00001:
        p = '>0.0001***'
        p = fbold(p) if bold else p
    elif p>0.000001:
        p = '>0.00001****'
        p = fbold(p) if bold else p
    else:
        p = '>0.000001!'
        p = fbold(p) if bold else p
    return p





# def plot_all_tables(table1, table2):
#     string1 = plot_table1(table1)
#     string2 = plot_table2(table2)    
#     file = os.path.join(report_dir, f'all_tables.{format}')
    
#     with open(file, 'w') as f:
#         f.write (('<br>'*5).join([css_format, string1, string2]))