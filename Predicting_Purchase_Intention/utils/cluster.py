from Predicting_Purchase_Intention.utils.model import initialize, pipeline_normalizer, initialize_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def cluster(X, df):

    threhsold_pca = 3
    pca = PCA(n_components=threhsold_pca, whiten=True)
    pca.fit(X)
    #instantiate and fit the PCA model
    X_proj = pd.DataFrame(pca.transform(X))

    kmeans_scaled = KMeans(n_clusters = 5)
    kmeans_scaled.fit(X_proj)

    labels_scaled = kmeans_scaled.labels_
    test_clusters = 5

    print('Working with ' + str(test_clusters) + ' clusters to segment the DataSet', flush=True)
    print("-"*80)

    model = KMeans(n_clusters = test_clusters, max_iter = 300)
    model.fit(X_proj)
    # labelling = model.labels_

    # test_labelled = pd.concat([df,pd.Series(labelling)],axis=1).rename(columns={0:"label"})
    # test_labelled = test_labelled.set_index('user_pseudo_id')

    # customer_mixes = {}

    # for cluster in np.unique(labelling):
    #     customer_mixes[cluster] = test_labelled[test_labelled.label == cluster]

    return model
