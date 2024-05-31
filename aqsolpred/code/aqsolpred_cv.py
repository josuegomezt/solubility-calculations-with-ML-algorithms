# -*- coding: utf-8 -*-
"""
Created on 04-03-2020 23:39:16 2020

@author:  Murat Cihan Sorkun

This code crossvalidates on test set. Each fold selected part of used as test and the remaning part included into training set
"""

import pandas as pd
import numpy as np 
import utilities
import datetime
import time
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import preprocess
from sklearn.model_selection import KFold

initial_start = time.time()

#Load Train/Test data 
train_set_df = pd.read_csv("../data/dataset-not-FA.csv")
test_set_df = pd.read_csv("../data/dataset-E.csv")


#Preprocess test and train sets
all_descriptors_train_df_all,element_list=preprocess.preprocess_train(train_set_df,test_set_df)
all_descriptors_test_df_all,test_smiles_list=preprocess.preprocess_test(test_set_df,element_list)
train_logS_list=all_descriptors_train_df_all["LogS"]
test_logS_list=all_descriptors_test_df_all["LogS"]

#Select descriptors by LASSO
selected_data_train, selected_data_test = preprocess.select_features_lasso(all_descriptors_train_df_all,all_descriptors_test_df_all)


full_test=[]
full_pred_mlp=[]
full_pred_xgb=[]
full_pred_rf=[]
full_pred_consensus=[]
full_smiles_list=[]


#Run cross validation over the test set
#for LOO ---> n_splits=len(selected_data_test)
i=0
kf = KFold(n_splits=4,shuffle=True,random_state=1)
for train, test in kf.split(selected_data_test):   
    
    
    start = time.time()
    i=i+1
    print("\n Fold " + str(i))
    kfold_data_test=selected_data_test.iloc[test,:]
    kfold_data_train=selected_data_test.iloc[train,:]
 
    kfold_logS_test=test_logS_list[test]
    kfold_logS_train=test_logS_list[train]
    
    merged_data_train=pd.concat([selected_data_train,kfold_data_train], axis=0)
    merged_logS=pd.concat([train_logS_list,kfold_logS_train], axis=0)
    
    #Train and Test with Neural Nets
    mlp_model = MLPRegressor(activation='tanh', hidden_layer_sizes=(500), max_iter=500, random_state=0, solver='adam')
    mlp_model.fit(merged_data_train, merged_logS)   
    pred_mlp = mlp_model.predict(kfold_data_test)
    utilities.get_errors(kfold_logS_test,pred_mlp,"Neural Nets")
    
    #Train and Test with XGBoost
    xgboost_model = xgboost.XGBRegressor(n_estimators=1000) 
    xgboost_model.fit(merged_data_train, merged_logS)    
    pred_xgb = xgboost_model.predict(kfold_data_test)
    utilities.get_errors(kfold_logS_test,pred_xgb,"XGBoost")
    
    #Train and Test with Random Forest
    rf_model = RandomForestRegressor(random_state=0, n_estimators=1000) 
    rf_model.fit(merged_data_train, merged_logS)   
    pred_rf = rf_model.predict(kfold_data_test)   
    utilities.get_errors(kfold_logS_test,pred_rf,"Random Forest")
    
    #Calculate Consensus results
    pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
    utilities.get_errors(kfold_logS_test,pred_consensus,"Consensus")
   
    #Append results of the fold
    full_test = full_test + list(kfold_logS_test)
    full_pred_mlp = full_pred_mlp + list(pred_mlp)
    full_pred_xgb = full_pred_xgb + list(pred_xgb)
    full_pred_rf = full_pred_rf + list(pred_rf)
    full_pred_consensus = full_pred_consensus + list(pred_consensus)
    full_smiles_list = full_smiles_list + list(map(test_smiles_list.__getitem__, test))
    
    time_counsumed = time.time()-start
    print("Time used:", time_counsumed)  
    
 
print("\n Full Results")       
utilities.get_errors(full_test,full_pred_mlp,"Neural Nets")
utilities.get_errors(full_test,full_pred_xgb,"XGBoost")
utilities.get_errors(full_test,full_pred_rf,"Random Forest")
utilities.get_errors(full_test,full_pred_consensus,"Consensus")


time_counsumed = time.time()-initial_start 
print("Total time used:", time_counsumed)  

#Create DF for results 
results=np.column_stack([full_pred_mlp,full_pred_xgb,full_pred_rf,full_pred_consensus,full_test])
df_results = pd.DataFrame(results, columns=['Neural Nets','Xgboost','Random Forest','Consensus','Target'])

#Calculate STD of 3 predictions
preds_all=df_results[['Neural Nets','Xgboost','Random Forest']].values
std=np.std(preds_all, axis = 1)
df_results["Residual"]=abs(df_results["Consensus"]-df_results["Target"])   
df_results["STD"]=std  
df_results["SMILES"]=full_smiles_list 

#Write results into a file
df_results=df_results.round(3)
df_results.to_csv("../results/results_cv4.csv",index=False)


