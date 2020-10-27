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
import pandas as pd
from statsmodels.stats.multitest import multipletests
sns.set(style='whitegrid')
### settings


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

############################
#####   CODE   #############
############################

def lineplot_table(table, title, columns=3, rows=None, save_to=None, 
                   xlabel=None, ylabel=None, n=-1):
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
            values = table[descriptor][group]['values']
            if len(values)==0:
                ax.clear()
                ax.text(0.5, 0.5, 'No Data', ha='center')
                break
            x = np.arange(values.shape[-1]) + (0 if group=='nt1' else 0.035*values.shape[-1])
            # mean values of feature
            y_mean = np.nanmean(values, 0)
            # upper std
            sem = stats.sem(values, 0, nan_policy='omit')
            # sem = np.std(values, 0)
            # lower std
            # err_kws = {'x':x, 'y1':y1, 'y2':y2, 'alpha':0.2, 'color':c[group]}
            # sns.pointplot(x=x, y=y_mean, ax=ax, c=c[group])
            ax.errorbar(x, y_mean, yerr=sem, c=c[group], fmt='-o', alpha=0.7)

        n_nt1 = table[descriptor]['nt1']['values'].shape[0]
        n_cnt = table[descriptor]['control']['values'].shape[0]
        # convert sleep stage to stage name if necessary
        if descriptor in [0,1,2,3,4,5]: 
            descriptor = cfg.num2stage[descriptor]
        if isinstance(descriptor, tuple):
            descriptor = '-'.join([str(cfg.num2stage[d]) for d in descriptor])
        if not isinstance(descriptor, str): 
            descriptor=str(descriptor)
        ax.set_title(descriptor + f' | values = {n_cnt+n_nt1} ({n_nt1}/{n_cnt})')
        ax.legend(['NT1', 'Control'])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    plt.suptitle(title + f' n = {n}', y=1)
    plt.pause(0.01)
    plt.tight_layout() 
    
    if save_to is None:
        save_to = os.path.join(cfg.documents, 'reports', f'{title}.png' )
    fig.savefig(save_to)
    return fig, axs


def distplot_table(table, title, columns=3, rows=None, save_to=None, 
                   xlabel=None, ylabel=None):
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
    
    if not isinstance(xlabel, list): xlabel=[xlabel]*len(table)
    if not isinstance(ylabel, list): ylabel=[ylabel]*len(table)
    assert len(xlabel) == len(table)
    assert len(ylabel) == len(table)

    n_plots = len(table)

    if rows is None:
        size = (int(np.ceil(n_plots/columns)), columns)
    if columns is None:
        size = (rows, int(np.ceil(n_plots/rows)))
              
    fig, axs = plt.subplots(*size)
        
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs] 
    
    for i, descriptor in enumerate(table): 
        ax = axs[i]
        values_nt1 = table[descriptor]['nt1'].get('values',[])
        values_cnt = table[descriptor]['control'].get('values',[])
        if sum(~np.isnan(values_nt1))==0 or sum(~np.isnan(values_cnt))==0: 
            ax.text(0.5, 0.5, 'No Data', ha='center')
            ax.set_title(descriptor)
            continue
        
        n_bins = min(15, max(len(np.unique(values_nt1)), len(np.unique(values_cnt))))
        
        vmin = min(np.nanmin(values_nt1), np.nanmin(values_cnt))
        vmax = max(np.nanmax(values_nt1), np.nanmax(values_cnt))
        
        bins = np.linspace(vmin, vmax, n_bins+1)
        second_ax = ax.twinx()
        try: 
            sns.distplot(values_nt1, bins=bins, ax=ax, norm_hist=False, kde=False)
            #Plotting kde without hist on the second Y axis
            sns.distplot(values_nt1, ax=second_ax, kde=True, hist=False)
            #Removing Y ticks from the second axis
            second_ax.set_yticks([])
        except Exception as e: print(e)  
        try: 
            sns.distplot(values_cnt, bins=bins, ax=ax, norm_hist=False, kde=False)
            #Plotting kde without hist on the second Y axis
            sns.distplot(values_cnt, ax=second_ax, kde=True, hist=False)
            #Removing Y ticks from the second axis
            second_ax.set_yticks([])
        except Exception as e: print(e)
        second_ax.set_ylim([0, second_ax.get_ylim()[1]*1.5])
                
        p_val = plotting.format_p_value(table[descriptor]['p'], bold=False)
        # convert sleep stage to stage name if necessary
        n_nt1 = np.sum(~np.isnan(table[descriptor]['nt1']['values']))
        n_cnt = np.sum(~np.isnan(table[descriptor]['control']['values']))
        if descriptor in [0,1,2,3,4,5]: 
            descriptor = cfg.num2stage[descriptor]
        if isinstance(descriptor, tuple):
            descriptor = '-'.join([str(cfg.num2stage[d]) for d in descriptor])
        if not isinstance(descriptor, str): descriptor=str(descriptor)

        ax.set_title(descriptor + f' - n = {n_cnt+n_nt1} ({n_nt1}/{n_cnt}) - p {p_val}')
        ax.legend(['NT1', 'Control'])
        ax.set_xlabel(xlabel[i])
        ax.set_ylabel(ylabel[i])
        if vmax<=1:
            ax.set_xlim(0,1)
        else:
            ax.set_xlim([vmin-(vmax-vmin)*0.05,  vmax+(vmax-vmin)*0.05])

        
        
    plt.suptitle(title + f' | n = {len(values_nt1)+len(values_cnt)}', y=1)
    plt.pause(0.01)
    plt.tight_layout() 
    
    if save_to is None:
        save_to = os.path.join(cfg.documents, 'reports', f'{title}.png' )
    fig.savefig(save_to)
    return fig, axs
        


def print_table(table, title, correction=False):
    """

    :param correction: the correction method to be applied, see statmodels.stats.multipletest.multipletests

    Format a dictionary as a table and save it to HTML/MD/CSV
    
    example dictionary
    dictionary['variable name']['group1/group2']['values'] = [0,5,2,3,4, ...]
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]
 
    """

    df = pd.DataFrame(columns=["Variable", "NT1", "Control", "p", f'p_corr_{correction}', 'effect size'])

    if correction:
        pvals = []
        for name, subtable in table.items():
                p = table[name]['p']
                pvals.append(0 if isinstance(p, str) else p)
        try:
            corr_pvals = multipletests(pvals, alpha=0.05, method=correction)
        except:
            corr_pvals =[ [np.nan for _ in pvals]]*2
        i=0
        for name, subtable in table.items():
                table[name][f'p_corr_{correction}'] = corr_pvals[1][i]
                i+=1

    for name, d in table.items():
        nt1_mean, nt1_std = d['nt1']['mean'], d['nt1']['std']
        c_mean, c_std = d['control']['mean'], d['control']['std']
        p = format_p_value(d['p'], bold=False)
        cohen_d = d['d']
        p_corr = format_p_value(d.get(f'p_corr_{correction}', '-'), bold=False)
        df.loc[len(df)] = [name, f'{nt1_mean:.2f} ± {nt1_std:.2f}', f'{c_mean:.2f} ± {c_std:.2f}', p, p_corr, cohen_d]

    
    string = df.to_html(escape=False, index_names=False)
    possible_plot = f'<br><br><br><br><img src="{title}.png" alt="not found: {title}.png">'

    html_file = os.path.join(report_dir, f'{title}.html')
    xlsx_file = os.path.join(report_dir, f'{title}.xlsx')

    with open(html_file, 'w') as f:
        f.write(css_format + string + possible_plot)

    df.to_excel(xlsx_file)
    return string

def print_table_with_subvars(table, title, correction=False):
    """

    :param correction: the correction method to be applied, see statmodels.stats.multipletest.multipletests

    Format a dictionary as a table and save it to HTML/MD/CSV
    
    example dictionary
    dictionary['variable name']['group1/group2']['values'] = [0,5,2,3,4, ...]
    dictionary['variable name']['group1/group2']['mean'] = ...
    dictionary['variable name']['group1/group2']['std'] = ...
    dictionary['variable name']['p']  = 0.05
    
    alternatively with subvars:
        dictionary['variable name']['subvarname']['group1/group2']['values'] = [0,5,2,3,4, ...]
 
    """

    df = pd.DataFrame(columns=["Variable",'Subvar', "NT1", "Control", "p", f'p_corr_{correction}', 'effect size'])

    if correction:
        pvals = []
        for name, subtable in table.items():
            for subvar in subtable:
                pvals.append(table[name][subvar]['p'])
        corr_pvals = multipletests(pvals, alpha=0.05, method=correction)
        i=0
        for name, subtable in table.items():
            for subvar in subtable:
                table[name][subvar][f'p_corr_{correction}'] = corr_pvals[1][i]
                i+=1


    for name, subtable in table.items():
        if name in [0,1,2,3,4,5]:
            name = cfg.num2stage[name]
        df.loc[len(df)] = [name, '', '', '', '', '', '']

        for subvar in subtable:
            nt1_mean, nt1_std = subtable[subvar]['nt1']['mean'], subtable[subvar]['nt1']['std']
            c_mean, c_std = subtable[subvar]['control']['mean'], subtable[subvar]['control']['std']
            p = format_p_value(subtable[subvar]['p'], bold=False)
            p_corr = format_p_value(subtable[subvar].get(f'p_corr_{correction}', '-'), bold=False)
            cohen_d = subtable[subvar]['d']
            if subvar in [0,1,2,3,4,5]:
                subvar = cfg.num2stage[subvar]
            if nt1_mean<0.1:
                pf = 4
            elif nt1_mean<1:
                pf = 3
            else:
                pf = 2
            df.loc[len(df)] = ['', subvar, f'{nt1_mean:.{pf}f} ± {nt1_std:.{pf}f}', f'{c_mean:.{pf}f} ± {c_std:.{pf}f}', p, p_corr, cohen_d]

    report_dir = os.path.join(cfg.documents, 'reports')
    html_file = os.path.join(report_dir, f'{title}.html')
    xlsx_file = os.path.join(report_dir, f'{title}.xlsx')

    string = df.to_html(escape=False, index_names=False)
    possible_plot = f'<br><br><br><br><img src="{title}.png" alt="not found: {title}.png">'
    with open(html_file, 'w') as f:
        f.write(css_format + string + possible_plot)
    df.to_excel(xlsx_file)
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
        p = '<0.001**'
        p = fbold(p) if bold else p
    elif p>0.00001:
        p = '<0.0001***'
        p = fbold(p) if bold else p
    elif p>0.000001:
        p = '<0.00001****'
        p = fbold(p) if bold else p
    else:
        p = '<0.000001!'
        p = fbold(p) if bold else p
    return p





# def plot_all_tables(table1, table2):
#     string1 = plot_table1(table1)
#     string2 = plot_table2(table2)    
#     file = os.path.join(report_dir, f'all_tables.{format}')
    
#     with open(file, 'w') as f:
#         f.write (('<br>'*5).join([css_format, string1, string2]))