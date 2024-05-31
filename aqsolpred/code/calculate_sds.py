#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 08 2019

@author: Murat Cihan Sorkun

SD Calculation of solubility datasets

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns

#read data file contains datapoints having minimum 2 occurences from all datasets   
multiple_data_desc = pd.read_csv("../data/multiple_data.csv", header=0)  

#Calculate deviation from mean value for each datapoint
solubility= multiple_data_desc['Solubility'].values
mean= multiple_data_desc['Mean'].values
residual=solubility-mean
abs_residual=np.abs(residual)


#######- Calculates deviation of each datapoint and groups based on source dataset -########

IdList  = multiple_data_desc['ID']
residual_counter_source=[0]*9 
residual_total_source=[0]*9
square_residual_total_source=[0]*9

for i in range(0,len(multiple_data_desc)): 
    counter=0
    if(IdList[i][0]=="A"):
        residual_total_source[0]=residual_total_source[0]+abs_residual[i]
        square_residual_total_source[0]=square_residual_total_source[0]+abs_residual[i]*abs_residual[i]
        residual_counter_source[0]=residual_counter_source[0]+1
    elif(IdList[i][0]=="B"):
        residual_total_source[1]=residual_total_source[1]+abs_residual[i]
        square_residual_total_source[1]=square_residual_total_source[1]+abs_residual[i]*abs_residual[i]
        residual_counter_source[1]=residual_counter_source[1]+1
    elif(IdList[i][0]=="C"):
        residual_total_source[2]=residual_total_source[2]+abs_residual[i]
        square_residual_total_source[2]=square_residual_total_source[2]+abs_residual[i]*abs_residual[i]
        residual_counter_source[2]=residual_counter_source[2]+1
    elif(IdList[i][0]=="D"):
        residual_total_source[3]=residual_total_source[3]+abs_residual[i]
        square_residual_total_source[3]=square_residual_total_source[3]+abs_residual[i]*abs_residual[i]
        residual_counter_source[3]=residual_counter_source[3]+1
    elif(IdList[i][0]=="E"):
        residual_total_source[4]=residual_total_source[4]+abs_residual[i]
        square_residual_total_source[4]=square_residual_total_source[4]+abs_residual[i]*abs_residual[i]
        residual_counter_source[4]=residual_counter_source[4]+1
    elif(IdList[i][0]=="F"):
        residual_total_source[5]=residual_total_source[5]+abs_residual[i]
        square_residual_total_source[5]=square_residual_total_source[5]+abs_residual[i]*abs_residual[i]
        residual_counter_source[5]=residual_counter_source[5]+1
    elif(IdList[i][0]=="G"):
        residual_total_source[6]=residual_total_source[6]+abs_residual[i]
        square_residual_total_source[6]=square_residual_total_source[6]+abs_residual[i]*abs_residual[i]
        residual_counter_source[6]=residual_counter_source[6]+1
    elif(IdList[i][0]=="H"):
        residual_total_source[7]=residual_total_source[7]+abs_residual[i]
        square_residual_total_source[7]=square_residual_total_source[7]+abs_residual[i]*abs_residual[i]
        residual_counter_source[7]=residual_counter_source[7]+1
    elif(IdList[i][0]=="I"):
        residual_total_source[8]=residual_total_source[8]+abs_residual[i]
        square_residual_total_source[8]=square_residual_total_source[8]+abs_residual[i]*abs_residual[i]
        residual_counter_source[8]=residual_counter_source[8]+1  


#Calculates standard deviation of each dataset (Equation-1 in the paper)        
residual_mean_source=np.divide(residual_total_source,residual_counter_source)
sd_source=np.sqrt(np.divide(square_residual_total_source,residual_counter_source))


#Creates dataframe with dataset names
source_names=["A","B","C","D","E","F","G","H","I"]
df_sd_errors = pd.DataFrame({'Dataset':source_names, 'SD (LogS)':sd_source,'Type':['SD']*9})
print(df_sd_errors)

#Save bar chart display for SD of each dataset
sns.set_style("whitegrid")
pyplot.figure(figsize=(15,9))
error_plot=sns.barplot(y='SD (LogS)', x='Dataset',data=df_sd_errors,hue='Type', palette = 'Blues') 
pyplot.savefig('../results/sd_datasets.png')

#Write results into a file
df_sd_errors.to_csv("../results/sd_dataset.csv",index=False)

