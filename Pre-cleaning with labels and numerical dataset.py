# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:15:17 2021

@author: kanta
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Loading the dataset with numerical data
data_file = 'ESS5_data.xlsx' 
df = pd.read_excel(data_file)
#Loading the dataset with the categorical data and the labels. It will be 
#needed in order to find the na values
labels = 'ESS5e03_4 (2).xlsx' 
df_labels = pd.read_excel(labels)

#Looking at the columns of the numerical dataset and the labels dataset
df.head()
df_labels.head()

#Replacing in the label dataset the column names with those of the numerical dataset
col_names = df.columns 
df_labels.columns = col_names 

#Setting the threshold of the age and age to retire 
agea_thresh = 39 
agertr_thresh = 39 

#Filtering out the rows that are not needed for the analysis from the numerical dataset
df_filtered = df[((df['agertr'] > agertr_thresh) &
                  (df['agea'] > agea_thresh)     &            
                  (df['agertr'] != 666)          &
                  (df['agertr'] != 777)          &
                  (df['agertr'] != 888)          &
                  (df['agertr'] != 999)          & 
                  (df['rtryr']>=6666)            &
                  (df['agea']<100))]

#Filtering out the rows that are not needed for the analysis from the labels dataset             
df_labels_rtr2 = df_labels[((df_labels['agertr'] != 'Refusal')   &
                  (df_labels['agertr'] != "Don\'t know")         &
                  (df_labels['agertr'] != 'Not applicable')      &
                  (df_labels['agertr'] != 'No answer'))          &
                  ((df_labels['rtryr'] == 'Not applicable')      |
                  (df_labels['rtryr'] == 'Refusal')              |
                  (df_labels['rtryr'] == "Don\'t know")          |
                  (df_labels['rtryr'] == 'No answer'))           &
                  (df_labels['agea'] !='Not available')] 

         
df_labels_rtr = df_labels_rtr2 [(df_labels_rtr2 ['agea'] > agea_thresh)  & 
                  (df_labels_rtr2 ['agea'] < 100)                          & 
                  (df_labels_rtr2 ['agertr'] > agertr_thresh)]

#Defining the nas as they are in the labels dataset and replacing with nas in
#numerical dataset                
NAs = ['Refusal', 'Don\'t know', 'Not applicable', 'No answer', 'Not available']
df_filtered[df_labels_rtr.isin(NAs)] = np.nan


#Saving the file to csv in order to have a less heavy and easy to load and work on file
df_filtered.to_csv('ESS5_with_NaNs_v2.csv', index=False)

