#%%
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from numpy import unique
from numpy import where
import matplotlib.pyplot as plt
import missingno as msno

#%%
## Some basic data preparation
synthetic_input_data_pd = synthetic_input_data.toPandas()
volvo_input_data_transformed_pd = volvo_input_data_transformed.toPandas()
synthetic_data_box = synthetic_input_data_pd.copy()
synthetic_data_box.drop(["spare_part_id", "yyyymm"], axis=1, inplace=True)
synthetic_data_box["demand_quantity"] = synthetic_data_box["demand_quantity"].astype('float')


#%%
# Checking if none columns doesn't conatain any outlier. Maybe the a column contains only data for one observation or the majority is one kind
part_count = synthetic_input_data_pd.spare_part_id.value_counts()
time_count = synthetic_input_data_pd.yyyymm.value_counts()
demand_count = synthetic_input_data_pd.demand_quantity.value_counts()
plt.plot(demand_count)
plt.figure(figsize=(16, 10))


#%%
# Checking if none columns doesn't conatain any outlier. Maybe the a column contains only data for one observation or the majority is one kind
volvo_input_data_transformed_df = volvo_input_data_transformed.toPandas()
v_part_count = volvo_input_data_transformed_df.spare_part_id.value_counts()
time_b_count = volvo_input_data_transformed_df.time_bucket.value_counts()
time_btype_count = volvo_input_data_transformed_df.time_bucket_type.value_counts()
demand_l_count = volvo_input_data_transformed_df.total_demand_lines.value_counts()
demand_q_count = volvo_input_data_transformed_df.total_demand_quantity.value_counts()
plt.plot(demand_q_count)
plt.figure(figsize=(16, 10))

#%%
# checking if we have any missing data
msno.matrix(synthetic_input_data_pd)
# checking if we have any missing data
msno.matrix(volvo_input_data_transformed_pd)

#%%
# Boxplotting
green_diamond = dict(markerfacecolor='g', marker='D')
fig3, ax3 = plt.subplots()
ax3.set_title('Changed Outlier Symbols')
ax3.boxplot(synthetic_data_box.demand_quantity, flierprops=green_diamond)

red_diamond = dict(markerfacecolor='r', marker='D')
fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.set_title('Changed Outlier Symbols')
ax3.boxplot(volvo_input_data_transformed_pd.total_demand_quantity, flierprops=red_diamond)

red_diamond = dict(markerfacecolor='r', marker='D')
fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.set_title('Changed Outlier Symbols')
ax3.boxplot(volvo_input_data_transformed_pd.total_demand_lines, flierprops=red_diamond)

#%%
X = synthetic_input_data_pd.copy()
X = X.drop(["spare_part_id", "yyyymm"], axis=1)

kmeans_alg = ["elkan", "auto", "full"]

#%%
# Isolation Forest
def fit_iforest(contamination, df):
    iso = IsolationForest(contamination=contamination)
    iso_pred = iso.fit_predict(df)
    
    return iso_pred
#%%


#%%
# Kmeans
def fit_kmeans(alg, df):
    
    kmeans = KMeans(n_clusters=2, algorithm=alg).fit(df)
    kmeans_pred = kmeans.labels_
    
    return kmeans_pred


X["is_outlier_if"] = X["is_outlier_if"].map({1:0, -1:1})
X["is_outlier_svm"] = X["is_outlier_svm"].map({1:0, -1:1})
X["is_outlier_lof"] = X["is_outlier_lof"].map({1:0, -1:1})
X["is_outlier_lof"].value_counts()



#%%
fit_lof_arr = []
fit_oc_svm_arr = []
fit_iforest_arr = []
fit_kmeans_arr = []


#%%
def predict_outliers(df):
    
    for alg in lof_alg: # 4 models
        fit_lof_arr.append(fit_lof(alg, df))
        
    for contamination in contaminations:
        fit_oc_svm_arr.append(fit_oc_svm(contamination, df)) # 5 models
        fit_iforest_arr.append(fit_iforest(contamination, df)) # 5 models
    
    for alg in kmeans_alg: # 3 models
        fit_kmeans_arr.append(fit_kmeans(alg, df).transpose())
    
    
    k_means_df = pd.DataFrame(fit_kmeans_arr)
    k_means_df.transpose()
    lof_df = pd.DataFrame(fit_lof_arr)
    lof_df.transpose()
    oc_svm_df = pd.DataFrame(fit_oc_svm_arr)
    oc_svm_df.transpose()
    iforest_df = pd.DataFrame(fit_iforest_arr)
    iforest_df.transpose()
    
    return k_means_df, lof_df, oc_svm_df, iforest_df

#%%
pred_out = predict_outliers(X)

#%%
# Compute the mean out of the predictions and consider as outliers only the points with more than 75%
def voting(row):  
    if row['voting_rate'] > 0.75:
        return 0
    else:
        return 1

#%%
def sum(row):
    rate = sum(row)
    
    return rate


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

    for spare_part in spare_parts:

        print(spare_part)
        part_df = df.where(df[id] == spare_part).dropna()
        part_df.drop(id, axis=1, inplace=True)

        part_df = StandardScaler().fit_transform(part_df.values.reshape(-1, 1))
        part_df = pd.DataFrame(part_df)

        
        dbscan_labels.extend(fit_dbscan(part_df, 3, 5))
        ifores_labels.extend(fit_iforest(part_df, 0.01))

        for k in ocsvm_kernels:
            if len(ocsvm_labels) < len(ocsvm_kernels):
                ocsvm_labels[k] = []
            else:
                ocsvm_labels[k].extend(fit_oc_svm(part_df, k))

        for l_alg in lof_algs:
            if len(lof_labels) < len(lof_algs):
                lof_labels[l_alg] = []
            else:
                lof_labels[l_alg].extend(fit_lof(part_df, l_alg))  


    df['dbscan_lbls'] = dbscan_labels
    df['iforest_lbls'] = ifores_labels


    return(df, ocsvm_labels)


# %%
def anomaly_detector(data_frame, dimensions=1, **params):


    if dimensions == 1: # the single dimensions flow
        
        full_df = get_labels(data_frame, 'spare_part_id', 1)

    elif dimensions >= 2: # multidimensional flow
        pass
    else:
        print("Garbage value for dimensions [it is expected to be 1 for 1D data and 2+D for multidimensional data]")


    return(data_frame)
# %%
