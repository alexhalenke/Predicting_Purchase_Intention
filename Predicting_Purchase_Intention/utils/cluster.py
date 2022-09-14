from Predicting_Purchase_Intention.utils.model import initialize, pipeline_normalizer, initialize_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import numpy as np
import pandas as pd


def cluster(X,y):
    scaler = pipeline_normalizer(X)[0]
    binary_col= pipeline_normalizer(X)[1]
    robust_scaling_cat = pipeline_normalizer(X)[2]
    standard_scaling_cat = pipeline_normalizer(X)[3]
    
    test_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    threhsold_pca = 3
    pca = PCA(n_components=threhsold_pca, whiten=True)
    pca.fit(test_scaled)
    test_proj = pd.DataFrame(pca.transform(test_scaled))
    
    kmeans_scaled = KMeans(n_clusters = 5)
    kmeans_scaled.fit(test_proj)
    
    labels_scaled = kmeans_scaled.labels_    
    test_clusters = 5

    print('Working with ' + str(test_clusters) + ' clusters to segment the DataSet', flush=True)
    print("-"*80)

    kmeans = KMeans(n_clusters = test_clusters, max_iter = 300)
    kmeans.fit(test_proj)
    labelling = kmeans.labels_
    
    test_labelled = pd.concat([test_df,pd.Series(labelling)],axis=1).rename(columns={0:"label"})
    test_labelled = test_labelled.set_index('user_pseudo_id')
    
    customer_mixes = {}

    for cluster in np.unique(labelling):
        customer_mixes[cluster] = test_labelled[test_labelled.label == cluster]
        
    
