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




# First, build a dataframe with the same explantion results as the baseline
# baseline: same_lag
def avg_lag(explained_dataset):
    l = len(explained_dataset.columns)
    list_ones = [1] * l                 # The values in the table are all 1
    data = pd.DataFrame(list_ones)
    data.index = explained_dataset.columns
    
    return data

# shap
def shap_results(model_name, explained_dataset):
    
    explainer = shap.TreeExplainer(model_name)
    shap_values = explainer(explained_dataset,check_additivity=False)
    
    shap_frame = pd.DataFrame(shap_values.values)
    df_global = pd.DataFrame()
    
    for i in range (0, len(shap_frame.columns)):
        shap_values_update = []
        each_shap_value = (abs(shap_frame[i]).sum())/len(shap_frame)
        
        shap_values_update.append(each_shap_value)
    
        df_global[i] = shap_values_update
        del shap_values_update
        
    df_global.columns = explained_dataset.columns
    df_global = df_global.T
        
    return df_global


# fi
def fi_results(model_name, explained_dataset):
    feature_importance = model_name.feature_importances_
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance.index = explained_dataset.columns
    
    return feature_importance

#fi-shap
def fi_shap_results(fi_explain, shap_explain):
    
    fi_sum = fi_explain[0].sum()
    fi_shap_list = []
    
    for n in range(0, len(shap_explain)):
        
        index_name = (list(shap_explain.index))[n]
        
        weight_by_fi = (fi_explain.loc[index_name,0]) / fi_sum
        fi_shap_values = shap_explain.loc[index_name,0] * weight_by_fi
        fi_shap_list.append(fi_shap_values)
    
    fi_shap_df = pd.DataFrame(fi_shap_list)
    fi_shap_df.index = shap_explain.index
    
    return fi_shap_df
