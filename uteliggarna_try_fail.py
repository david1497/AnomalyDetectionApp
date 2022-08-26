#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# %%
data_file = "synthetic_input_data.csv"
data_set = pd.read_csv(data_file, sep=";")
true_labels = pd.read_csv("syntethic_original.csv", sep=";")
true_labels = true_labels.iloc[:19056, -5:]
true_labels = true_labels.iloc[:, :1]


#%%
def fit_dbscan(X, eps, min_samples):
    '''
    This function builds the DBSCAN model and returns the labels(clusters) for each row of data.
    It takes as parameters X - which is the data set that should be clustered (labeled)
    eps or epsilor, which is a very critical parameter for the model
    min_samples - The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
    '''
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = dbscan_model.labels_

    return(labels)


#%%
def fit_iforest(X, contamination, bootstrap=True):
    """
    
    """
    iforest_model =  IsolationForest(contamination=contamination, bootstrap=bootstrap)
    labels = iforest_model.fit_predict(X)

    return(labels)


#%%
# One-Class SVM
ocsvm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

def fit_oc_svm(X, k):
    """
    """
    oc_svm_model = OneClassSVM(kernel=k)
    labels = oc_svm_model.fit_predict(X)
    
    return(labels)


#%%
# LOF
lof_algs = ['auto', 'ball_tree', 'kd_tree', 'brute']

def fit_lof(X, alg):
    """
    """
    lof_model = LocalOutlierFactor(algorithm=alg)
    lof_pred = lof_model.fit_predict(X)
    
    return lof_pred


#%%
def election(v, n_voters):
    voters = v
    
    ax = voters.replace(1, 0) # replacing inlier with 0
    am = ax.replace(-1, 1) # replacing outlier with 1
    sum_votes = am.sum(axis='columns') # summing votes across voters
    election_results = sum_votes / n_voters
    
    return(election_results)



#%%
def get_labels(df, id, n_features):
    """ 
    df - data frame to be used
    id - the column which contains the spare parts
    n_features - number of features in the dataset
    """
    spare_parts = df[id].drop_duplicates()

    ifores_labels = []
    dbscan_labels = []
    ocsvm_labels = {}
    lof_labels = {}
    
    
    if len(ocsvm_labels) < len(ocsvm_kernels):    
        for k in ocsvm_kernels:
            ocsvm_labels[k] = []
        for l_alg in lof_algs:
            lof_labels[l_alg] = []
    
    for spare_part in spare_parts:

        #print(spare_part)
        part_df = df.where(df[id] == spare_part).dropna()
        part_df.drop(id, axis=1, inplace=True)
        part_df.drop('yyyymm', axis=1, inplace=True)

        #part_df = StandardScaler().fit_transform(part_df.values.reshape(-1, 1))
        #part_df = pd.DataFrame(part_df)
        
        
        dbscan_labels.extend(fit_dbscan(part_df, 3, 5))
        ifores_labels.extend(fit_iforest(part_df, 0.01))

        for k in ocsvm_kernels:
            ocsvm_labels[k].extend(fit_oc_svm(part_df, k))

        for l_alg in lof_algs:
            lof_labels[l_alg].extend(fit_lof(part_df, l_alg))  

            
    for k in ocsvm_kernels:
        df[f'ocsvm_{k}'] = ocsvm_labels[k]
    for l_alg in lof_algs:
        df[f'lof_{l_alg}'] = lof_labels[l_alg]
    df['dbscan_lbls'] = dbscan_labels
    df['iforest_lbls'] = ifores_labels


    return(df)


# %%
def anomaly_detector(data_frame, dimensions=1, **params):

    initial_shape = data_frame.shape[1]
    
    if dimensions == 1: # the single dimensions flow

        full_df = get_labels(data_frame, 'spare_part_id', 1)
    
    elif dimensions >= 2: # multidimensional flow
        pass
    else:
        print("Garbage value for dimensions [it is expected to be 1 for 1D data and 2+D for multidimensional data]")
    
    final_shape = full_df.shape[1]
    new_columns = final_shape - initial_shape
    
    voters = full_df.iloc[:, new_columns:]
    
    data_frame = data_frame.iloc[:, :initial_shape]
    data_frame['labels'] = election(voters, new_columns)
    data_frame['labels'] = data_frame['labels'].apply(lambda x: 1 if x >= 0.7 else 0)
    
    return(data_frame)


# %%
test_phase, full_voters = anomaly_detector(data_set)
test_phase['true_labels'] = true_labels


#%%
final_df = test_phase.iloc[:, :3]
amx = test_phase.iloc[:, -2:]
final_df = final_df.merge(amx, how="left", left_index=True, right_index=True)


#%%
confusion_matrix(final_df['true_labels'], final_df['labels'])


#%%
votes = test_phase.groupby(['labels'])['labels'].count()