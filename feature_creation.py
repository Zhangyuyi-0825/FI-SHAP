import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Fuse dataframe and create time features
def mergeDataframe(set_df, add_dataset, feature_name):
    for i in range(0,len(set_df)):
        set_df[i] = pd.merge(set_df[i],add_dataset,how = 'outer',on = [feature_name])
        set_df[i] = set_df[i].dropna(axis = 0, how = 'any')
        
        set_df[i]['Year'] = pd.to_datetime(set_df[i]['DATE_TIME']).dt.year
        set_df[i]['Month'] = pd.to_datetime(set_df[i]['DATE_TIME']).dt.month
        set_df[i]['Day'] = pd.to_datetime(set_df[i]['DATE_TIME']).dt.day
        set_df[i]['Hour'] = pd.to_datetime(set_df[i]['DATE_TIME']).dt.hour
        set_df[i]['Minute'] = pd.to_datetime(set_df[i]['DATE_TIME']).dt.minute
        
    return set_df


#Create regular lag variable's names
def build_df_name(source_key_list,name):
    createDataframe = locals()
    dataframeList = []
    for i in range(0,len(source_key_list)):
        createDataframe[name + str(i)] = 'lag{}' + name + str(i)
        dataframeList.append(createDataframe[name + str(i)])
    return dataframeList



# Build dataframes with lagged variables
# exp_results must be the explanation results of continuous feature arranged vertically (continuous_feature)
# orig_dataset: The original dataset that needs to be augmented
# exp_results: explanation results produced by different methods
# iterations: the total number of features you want to build (this is a variable that can be set larger if the computer's computing power allows)

def improve_process(orig_dataset, exp_results, iterations):
    
    #lag_number = []
    #feature_name = []
    
    sum_values = float(exp_results.iloc[:,0].sum())  # the sum of explanation results
    sort_exp_results = pd.DataFrame(exp_results[0].sort_values(ascending=False, inplace=False)) # Sort features by explanation value (descending order)
    
    for i in range(0, len(sort_exp_results)):
        
        feature_name = list(sort_exp_results.index)   # feature's name
        feature_values = float(sort_exp_results.iloc[i,0])      # the explanation of features
        number = int((feature_values/sum_values) * iterations)  # number of lag periods required
        #lag_number.append(number)
                    
        if number <= 1:   
            break         # If the number of periods to lag is less than or equal to 1, stop(this is why it is necessary to sort the features)
        else:             # If the number of periods is greater than 1, you can start creating lagged variables 
            data_lag = pd.DataFrame(orig_dataset[feature_name[i]].copy())            # Create intermediate transition dataframe for lag variables
            data_lag.columns = [feature_name[i]]                                     # Make sure the feature's names are consistent
            
            lag_name = build_df_name(list(sort_exp_results.index), 'feature')        # Create the feature's name of the "lag variable"
            lag_feature_name = lag_name[i]                                           # Select feature's name in order
            
            for n in range(0, (number+1)):
                data_lag[lag_feature_name.format(n)] = data_lag[feature_name[i]].shift(n)         # dataframe with lag feature
            
            data_lag_finish = data_lag.drop(columns = [feature_name[i]])                          # Eliminate the original features to avoid duplication in the final merge
        
            orig_dataset = pd.concat([orig_dataset,data_lag_finish],axis = 1)                     # Merge dataframe
            orig_dataset = orig_dataset.dropna(axis = 0, how = 'any')                             # Remove Nan values
            orig_dataset = orig_dataset.reset_index(drop=True)                                    # reset index
        
    return orig_dataset