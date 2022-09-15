from Predicting_Purchase_Intention.utils.model import initialize, pipeline_normalizer, initialize_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import numpy as np
import pandas as pd


def cluster(X,df):
    scaler = pipeline_normalizer(X)[0]
    binary_col= pipeline_normalizer(X)[1]
    robust_scaling_cat = pipeline_normalizer(X)[2]
    standard_scaling_cat = pipeline_normalizer(X)[3]
    #pipeline import to scale features of the X
    
    col_drop = []
    for index, num in enumerate(X.isnull().sum().sort_values(ascending = False)):
        if num > 0.95*len(X):
            col_drop.append(X.isnull().sum().sort_values(ascending = False).keys()[index])
    
    #drop columns that have a lot of missing values 
    X = X.drop(columns = col_drop)
    #scale the X with the pipeline from utils
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    threhsold_pca = 3
    #PCA will collapse the features into a three value for each row
    pca = PCA(n_components=threhsold_pca, whiten=True)
    #fit pca on the scaled X
    pca.fit(X_scaled)
    #instantiate and fit the PCA model
    X_proj = pd.DataFrame(pca.transform(X_scaled))
    
    kmeans_scaled = KMeans(n_clusters = 5)
    kmeans_scaled.fit(X_proj)
    #generate the labels for the kmeans test
    labels_scaled = kmeans_scaled.labels_    
    test_clusters = 5

    print('Working with ' + str(test_clusters) + ' clusters to segment the DataSet', flush=True)
    print("-"*80)
    #instantiate the Kmeans model
    model = KMeans(n_clusters = test_clusters, max_iter = 300)
    model.fit(X_proj)
    labelling = model.labels_
    
    test_labelled = pd.concat([df,pd.Series(labelling)],axis=1).rename(columns={0:"label"})
    test_labelled = test_labelled.set_index('user_pseudo_id')
    
    customer_mixes = {}

    for cluster in np.unique(labelling):
        customer_mixes[cluster] = test_labelled[test_labelled.label == cluster]
    #use the above code if you want to return a dataframe of some sort: customer_mixes[0] for example
    
    return model
        
    
def preproc_test(X):
    scaler = pipeline_normalizer(X)[0]
    binary_col= pipeline_normalizer(X)[1]
    robust_scaling_cat = pipeline_normalizer(X)[2]
    standard_scaling_cat = pipeline_normalizer(X)[3]
    #pipeline import to scale features of the X
    
    col_drop = []
    for index, num in enumerate(X.isnull().sum().sort_values(ascending = False)):
        if num > 0.95*len(X):
            col_drop.append(X.isnull().sum().sort_values(ascending = False).keys()[index])
    
    #drop columns that have a lot of missing values 
    X = X.drop(columns = col_drop)

    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    threhsold_pca = 3
    pca = PCA(n_components=threhsold_pca, whiten=True)
    pca.fit(X_scaled)
    #instantiate and fit the PCA model
    X_proj = pd.DataFrame(pca.transform(X_scaled))
    
    return X_proj