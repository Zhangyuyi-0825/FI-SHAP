import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Create enough dataframe's names
def build_df_name(source_key_list,name):
    createDataframe = locals()
    dataframeList = []
    for i in range(0,len(source_key_list)):
        createDataframe[name + str(i)] = 'df' + name + str(i)
        dataframeList.append(createDataframe[name + str(i)])
    return dataframeList

#Split the dataframe according to the specified feature
def build_df(orig_dataset, source_key_list, feature_name):
    
    name_list = build_df_name(source_key_list, feature_name)
    
    for n in range(0, len(name_list)):
        name_list[n] = pd.DataFrame()
        name_list[n] = orig_dataset[orig_dataset[feature_name] == source_key_list[n]]
        name_list[n].drop(feature_name,1,inplace=True)
        
    return name_list