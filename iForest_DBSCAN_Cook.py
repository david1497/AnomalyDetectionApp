#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest 


#%%
def load_data():
    housing = fetch_openml(name="house_prices", as_frame=True)
    data = housing['data']
    data['SalePrice'] = housing['target']
    return data, housing

def plot_outliers(data, cols, target_col, outlier_col):
    # linear predictive features:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    mapdict = {0: [0,0], 1:[0,1], 2:[1,0], 3:[1,1]}

    for n, c in enumerate(cols):
        axs[mapdict[n][0], mapdict[n][1]].scatter(data[c], data[target_col], c=data[outlier_col], cmap='plasma', vmin=0, vmax=1.5)
        axs[mapdict[n][0], mapdict[n][1]].set_xlabel(c)

    axs[0,0].set_ylabel(target_col)
    axs[1,0].set_ylabel(target_col)

    plt.show()

data, housing = load_data()

# %% Cooks distance
def model_test(data, cols, target_col, calc_cooks=True):
    y = data[target_col]
    x = data[cols].fillna(0)
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit() 

    if calc_cooks:
        np.set_printoptions(suppress=True)
        influence = model.get_influence()
        cooks = influence.cooks_distance
        data['CooksDistance'] = cooks[0]
        plt.figure(figsize=(9,5))
        plt.scatter(data['Id'], cooks[0], s=10)
        plt.xlabel('Id')
        plt.ylabel('Cook\'s Distance')
        plt.show()
    
    data['Prediction'] = model.predict(x)

    RMSE = np.sqrt(mean_squared_error(data[target_col], data['Prediction']))
    r2 = r2_score(data[target_col], data['Prediction'])

    print(f'RMSE: {RMSE:.0f}')
    print(f'r2: {r2:.4f}')

    if calc_cooks:
        display(data[cols + ['CooksDistance', target_col, 'Prediction']].sort_values(by='CooksDistance', ascending=False).head(30))
    return model, data


# all data
cols = ['GrLivArea', 'YearBuilt', 'LotArea', 'OverallQual']
target_col = 'SalePrice'
model, data = model_test(data, cols, target_col=target_col)

# exclude data points with a Cook's distance higher than the mean Cook's distance * 3
cdm3 = data['CooksDistance'].mean() * 3
new_data = data[data['CooksDistance'] < cdm3]
data['CooksOutliers'] = np.where(data['CooksDistance'] < cdm3, 0, 1)
print(f'N removed = {len(data) - len(new_data)}')
# run again to see the change in prediction
model, new_data = model_test(new_data, cols, target_col='SalePrice', calc_cooks=False)

# plot outliers
plot_outliers(data, cols, target_col='SalePrice', outlier_col='CooksOutliers')

# %% DBSCAN
def plot_k_distance_graph(df):
    nn = NearestNeighbors(n_neighbors=2)
    nbrs = nn.fit(df)
    distances, indices = nbrs.kneighbors(df)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.figure(figsize=(12,6))
    plt.plot(distances)
    plt.title('K-distance graph', fontsize=20)
    plt.xlabel('Data points sorted by distance', fontsize=14)
    plt.ylabel('Epsilon', fontsize=14)
    plt.show()


def run_dbscan(df, data, cols, target_col='SalePrice', eps=8000, min_samples=6, scaled=''):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    print(dbscan)

    dbs = dbscan.fit(df)
    clusters = dbs.labels_
    data[f'cluster{scaled}'] = clusters
    data[f'no_cluster{scaled}'] = np.where(data[f'cluster{scaled}'] == -1, 1, 0)

    print(len(data[data[f'cluster{scaled}'] == -1]))
    print(data[f'cluster{scaled}'].unique())
    return data


# scale data
cols = ['GrLivArea', 'YearBuilt', 'LotArea', 'OverallQual']
target_col = 'SalePrice'
df = data[cols + [target_col]]
df_scaled = StandardScaler().fit_transform(df)

# calculate epsilon using the K-distance graph
plot_k_distance_graph(df_scaled)

# run DBSCAN and return the data with cluster columns
data = run_dbscan(df_scaled, data, cols, eps=1.1, min_samples=6, scaled='_scaled')

plot_outliers(data, cols, target_col, outlier_col='no_cluster_scaled')


#%% isolation forest
def run_isolation_forest(df, data, contamination=0.025, max_features=2):
    isof = IsolationForest(random_state=30, contamination=contamination, max_features=max_features).fit(df)
    results = isof.predict(df)
    data['isolation scores'] = results
    data['outlier'] = np.where(data[f'isolation scores'] == -1, 1, 0)

    print(len(data[data['isolation scores'] == -1]))
    print(data[f'isolation scores'].unique())

    return data

data = run_isolation_forest(pd.DataFrame(df_scaled, columns=[cols+['SalePrice']]), data)
plot_outliers(data, cols, target_col='SalePrice', outlier_col='outlier')
