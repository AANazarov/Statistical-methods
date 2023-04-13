# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:37:51 2023

@author: Alexander A. Nazarov
"""

###############################################################################
#           Approximation. Part 1
#------------------------------------------------------------------------------
#           Preparation of input data
###############################################################################


import time
start_time = time.time()

%clear    # console cleanup command for IDE Spyder


#==============================================================================
#               CONNECTING MODULES AND LIBRARIES
#==============================================================================


# Standard modules and libraries
#-------------------------------

import os    # Miscellaneous operating system interfaces
import sys    # System-specific parameters and functions
import platform    # Access to underlying platform’s identifying data¶

import math
# connect all content of the math module, use without aliases
from math import *

import numpy as np
import scipy as sci
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the accuracy
DecPlace = 4    # number of decimal places - число знаков после запятой

# Numpy settings
np.set_printoptions(precision = DecPlace, 
                    floatmode='fixed')

# Pandas settings
pd.set_option('display.max_colwidth', None)
'''text in the cell was reflected completely regardless of length'''
#pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.float_format', lambda x: ('%.'+str(DecPlace)+'f') % x)

# Seaborn settings
sns.set_style("darkgrid")
sns.set_context(context='paper', font_scale=1, rc=None)
'''paper', 'notebook', 'talk', 'poster', None'''

# Mathplotlib settings
f_size = 8
plt.rcParams['figure.titlesize'] = f_size + 12
plt.rcParams['axes.titlesize'] = f_size + 10
plt.rcParams['axes.labelsize'] = f_size + 6
plt.rcParams['xtick.labelsize'] = f_size + 4
plt.rcParams['ytick.labelsize'] = f_size + 4
plt.rcParams['legend.fontsize'] = f_size + 6
plt.rcParams['text.usetex'] = False    # support for mathematical formulas in TeX


# Custom modules and libraries
#-----------------------------
sys.path.insert(1, 'D:\SKILL FACTORY\REPOSITORY\MyModulePython')
from my_module__stat import *


# Other settings
#---------------
Text1 = os.getcwd()    # current directory path
#print(f'Сurrent directory path: {Text1} \n')


#==============================================================================
#               CONSTANTS
#==============================================================================

NumberCharLine = 79    # string number
TableHead1 = ('#' * NumberCharLine)
TableHead2 = ('=' * NumberCharLine)
TableHead3 = ('-' * NumberCharLine)

# Constant controlling the output mode
'''
ConstWriteFile = True - output as a text file
ConstWriteFile = False - printing out to the console
'''
ConstWriteFile = False
# after the ConstWriteFile value changes, you must restart the Spyder kernel

# Output file name
FileName = 'Preparation of input data'



#==============================================================================
#------------------------------------------------------------------------------
#               MAIN
#------------------------------------------------------------------------------
#==============================================================================

# Select output mode (console/file)
#----------------------------------

if ConstWriteFile:
    sys.stdout = open(FileName + '.txt','w', \
                      encoding='utf-16')


#==============================================================================
#               SOURCE DATA
#==============================================================================

print(TableHead2)
Title = 'SOURCE DATA'
print(' ' * int((NumberCharLine - len(Title))/2), Title)
print(TableHead2, 2*'\n')

fuel_df = pd.read_csv(filepath_or_buffer='data/FUEL.csv', sep=';')
data_df = fuel_df.copy()    # create a copy of the original Dataframe
#print(f'data_df = \n\n {data_df} \n')
print('data_df =', 2*'\n', data_df, '\n')
data_df.info()
print('\n')

print('Descriptive raw data statistics:', 2*'\n', data_df.describe())
print(2*'\n')


# DATA PROCESSING
#----------------

print(TableHead3)
Title = 'DATA PROCESSING'
print(' ' * int((NumberCharLine - len(Title))/2), Title)
print(TableHead3, 2*'\n')


# 1. Detection of missed values and their removal
#------------------------------------------------

Title = '1. Detection of missed values and their removal'
print('-' * int((NumberCharLine - len(Title))/2), \
      Title, \
      '-' * int((NumberCharLine - len(Title))/2), 2*'\n')

# Visualization of missing values
print('Visualization of missing values:', '\n')
result_df, detection_values_df = \
    df_detection_values(data_df, detection_values=[nan, 0])
print( '\n', result_df, '\n')

# Excludes missing values from dataset
print('Excludes missing values from dataset:', '\n')

drop_labels = []    # list of rows to be deleted
for elem in detection_values_df.index:
    if detection_values_df.loc[elem].any():
        drop_labels.append(elem)

data_df = data_df.drop(index=drop_labels)

result_df, detection_values_df = \
    df_detection_values(data_df, detection_values=[nan, 0])
print( '\n', result_df, '\n')


# 2. Convert feature-dates to datetime format
#--------------------------------------------

Title = '2. Convert feature-dates to datetime format'
print('-' * int((NumberCharLine - len(Title))/2), \
      Title, \
      '-' * int((NumberCharLine - len(Title))/2), 2*'\n')

# Convert date from Excel format to datetime format:
data_df['Month'] = pd.to_datetime(
    data_df['Month'],
    dayfirst=True,
    origin='1900-01-01',
    unit='D')

# Move the date to the end of the month
data_df['Month'] = data_df['Month'] + \
    pd.tseries.offsets.DateOffset(months=1) + \
        pd.tseries.offsets.DateOffset(days=-3)

# Rename the column
data_df.rename(columns={'Month': 'Date'}, inplace=True)

print('data_df =', 2*'\n', data_df, '\n')
data_df.info()
print('\n')


# DATA SAVEING
#----------------

print(TableHead3)
Title = 'DATA SAVEING'
print(' ' * int((NumberCharLine - len(Title))/2), Title)
print(TableHead3, 2*'\n')

# Fuel Flow (liters per 100 km)
Y = np.array(data_df['FuelFlow'])
print(f'Fuel Flow (liters per 100 km): \n\n Y = \n {Y} \n {type(Y)} {len(Y)}')
print('\n')

# Mileage (km)
X1 = np.array(data_df['Mileage'])
print(f'Mileage (km): \n\n X1 = \n {X1} \n {type(X1)} {len(X1)}')
print('\n')

# Temperature (degrees celsius)
X2 = np.array(data_df['Temperature'])
print(f'Temperature (degrees celsius): \n\n X2 = \n {X2} \n {type(X2)} {len(X2)}')
print('\n')

# Date
Date = np.array(data_df['Date'])
print(f'Date: \n\n Date = \n {Date} \n {type(Date)} {len(Date)}')
print('\n')

# Forming a DataFrame from variables X1, X2, Y

dataset_df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'Y': Y})
print('dataset_df =', 2*'\n', dataset_df, '\n')
dataset_df.info()
print('\n')

# Save data to csv-file
dataset_df.to_csv(
    path_or_buf='data/dataset_df.csv',
    mode='w+',
    sep=';',
    index_label='Number')

# Save data to xlsx-file
dataset_df.to_excel(
    excel_writer='data/dataset_df.xlsx',
    sheet_name='data')


#==============================================================================
#               VISUALIZATION
#==============================================================================

print(TableHead2)
Title = 'VISUALIZATION'
print(' ' * int((NumberCharLine - len(Title))/2), Title)
print(TableHead2, 2*'\n')

# General project title
Task_Project = "Analysis of fuel consumption of a car"

# A title that captures a moment in time
AsOfTheDate = ""

# Title of the project section
Task_Theme = ""

# General project title for graphs
Title_String = f"{Task_Project}\n{AsOfTheDate}"

# Variable names
Variable_Name_T_month = "Monthly data"
Variable_Name_Y = "FuelFlow (liters per 100 km)"
Variable_Name_X1 = "Mileage (km)"
Variable_Name_X2 = "Temperature (degrees celsius)"

# Bounds of values of variables (when constructing graphs)
(X1_min_graph, X1_max_graph) = (0, 3000)
(X2_min_graph, X2_max_graph) = (-20, 25)
(Y_min_graph, Y_max_graph) = (0, 20)

# Fuel Flow (liters per 100 km)
graph_lineplot_sns(
    Date, Y,
    Ymin_in=Y_min_graph, Ymax_in=Y_max_graph,
    color='orange',
    title_figure=Title_String, #title_figure_fontsize=14,
    title_axes='FuelFlow',
    x_label=Variable_Name_T_month,
    y_label=Variable_Name_Y,
    label_legend='data')

# Mileage (km)
graph_lineplot_sns(
    Date, X1,
    Ymin_in=X1_min_graph, Ymax_in=X1_max_graph,
    color='grey',
    title_figure=Title_String, #title_figure_fontsize=14,
    title_axes='Mileage',
    x_label=Variable_Name_T_month,
    y_label=Variable_Name_X1,
    label_legend='data')

# Temperature (degrees celsius)
graph_lineplot_sns(
    Date, X2,
    Ymin_in=X2_min_graph, Ymax_in=X2_max_graph,
    color='cyan',
    title_figure=Title_String, #title_figure_fontsize=14,
    title_axes='Temperature',
    x_label=Variable_Name_T_month,
    y_label=Variable_Name_X2,
    label_legend='data')

# Fuel consumption ratio (Y) to mileage (X1)
graph_scatterplot_sns(
    X1, Y,
    Xmin=X1_min_graph, Xmax=X1_max_graph,
    Ymin=Y_min_graph, Ymax=Y_max_graph,
    color='orange',
    title_figure=Title_String, title_figure_fontsize=14,
    title_axes='Fuel consumption ratio (Y) to mileage (X1)', title_axes_fontsize=16,
    x_label=Variable_Name_X1,
    y_label=Variable_Name_Y,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    s=80)

# Fuel consumption ratio (Y) to average monthly temperature (X2)
graph_scatterplot_sns(
    X2, Y,
    Xmin=X2_min_graph, Xmax=X2_max_graph,
    Ymin=Y_min_graph, Ymax=Y_max_graph,
    color='orange',
    title_figure=Title_String, title_figure_fontsize=14,
    title_axes='Fuel consumption ratio (Y) to average monthly temperature (X2)',
    title_axes_fontsize=16,
    x_label=Variable_Name_X2,
    y_label=Variable_Name_Y,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    s=80)



    






