import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", context={'axes.labelsize':18,
                                'xtick.labelsize':15,
                                'ytick.labelsize':15})
import warnings
warnings.filterwarnings('ignore')

import datetime as dt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
import catboost as cab
from catboost import CatBoostRegressor
import shap



# Select the maximum and minimum values of the predicted results
def sort_and_index(list_name):
    max_index = list_name.index(max(list_name))
    min_index = list_name.index(min(list_name))
    
    max_values = float(list_name[max_index])
    min_values = float(list_name[min_index])
    
    return print("max is: ",max_values,"index is: ",max_index," ; ","min is: ",min_values,"index is: ",min_index)


def xgboost_results(set_df, number):
    acc = []
    rmse = []
    
    for i in range(0, len(set_df)):
        train_set = set_df[i][set_df[i]['DATE_TIME']<'2020-06-09 00:00']
        test_set  = set_df[i][set_df[i]['DATE_TIME']>='2020-06-09 00:00']
        
        x_train = train_set.drop(columns={'DATE_TIME','DAILY_YIELD','TOTAL_YIELD'},axis=1)
        y_train = train_set.loc[:,['DAILY_YIELD']]
        x_test = test_set.drop(columns={'DATE_TIME','DAILY_YIELD','TOTAL_YIELD'},axis=1)
        y_test = test_set.loc[:,['DAILY_YIELD']]
        
        model_xgb = XGBRegressor()
        
        model_xgb.fit(
            x_train, 
            y_train, 
            eval_metric="rmse", 
            eval_set=[(x_train, y_train),(x_test,y_test)], 
            verbose=False, 
            early_stopping_rounds = 100)
        
        prediction = model_xgb.predict(x_test)
        r2 = r2_score(y_test,prediction)
        RMSE = np.sqrt(mean_squared_error(y_test,prediction))
            
        acc.append(r2)
        rmse.append(RMSE)
    
    return acc, rmse 


def lightgbm_results(set_df, number):
    acc = []
    rmse = []
    
    for i in range(0, len(set_df)):
        train_set = set_df[i][set_df[i]['DATE_TIME']<'2020-06-09 00:00']
        test_set  = set_df[i][set_df[i]['DATE_TIME']>='2020-06-09 00:00']
        
        x_train = train_set.drop(columns={'DATE_TIME','DAILY_YIELD','TOTAL_YIELD'},axis=1)
        y_train = train_set.loc[:,['DAILY_YIELD']]
        x_test = test_set.drop(columns={'DATE_TIME','DAILY_YIELD','TOTAL_YIELD'},axis=1)
        y_test = test_set.loc[:,['DAILY_YIELD']]
        
        model_lgb = lgb.LGBMRegressor()
        
        model_lgb.fit(x_train, y_train,eval_metric="rmse", 
                            eval_set=[(x_train, y_train)], 
                            verbose=False, 
                            early_stopping_rounds = 100)
        
        prediction = model_lgb.predict(x_test)
        r2 = r2_score(y_test,prediction)
        RMSE = np.sqrt(mean_squared_error(y_test,prediction))
            
        acc.append(r2)
        rmse.append(RMSE)
    
    return acc, rmse 

def catboost_results(set_df, number):
    acc = []
    rmse = []
    
    for i in range(0, len(set_df)):
        train_set = set_df[i][set_df[i]['DATE_TIME']<'2020-06-09 00:00']
        test_set  = set_df[i][set_df[i]['DATE_TIME']>='2020-06-09 00:00']
        
        x_train = train_set.drop(columns={'DATE_TIME','DAILY_YIELD','TOTAL_YIELD'},axis=1)
        y_train = train_set.loc[:,['DAILY_YIELD']]
        x_test = test_set.drop(columns={'DATE_TIME','DAILY_YIELD','TOTAL_YIELD'},axis=1)
        y_test = test_set.loc[:,['DAILY_YIELD']]
        
        model_cat = CatBoostRegressor(eval_metric='RMSE')

        model_cat.fit(x_train, y_train, 
                        eval_set=[(x_train, y_train)], 
                        verbose=False, 
                        early_stopping_rounds = 100)
        
        prediction = model_cat.predict(x_test)
        r2 = r2_score(y_test,prediction)
        RMSE = np.sqrt(mean_squared_error(y_test,prediction))
            
        acc.append(r2)
        rmse.append(RMSE)
    
    return acc, rmse   
