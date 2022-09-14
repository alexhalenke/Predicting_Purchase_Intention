from Predicting_Purchase_Intention.utils.params import LOCAL_DATA_PATH, SOURCE
from Predicting_Purchase_Intention.utils.clean_data import clean_data, drop_cols, expand_raw_cols
from Predicting_Purchase_Intention.utils.data import get_pandas, save_pandas
from Predicting_Purchase_Intention.utils.preprocess import put_together
from Predicting_Purchase_Intention.utils.model import pipeline_normalizer
from Predicting_Purchase_Intention.utils.registry import save_model, load_model
from Predicting_Purchase_Intention.utils.cluster import cluster
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import glob

import pickle

def preprocess():

    if SOURCE=='processed/':
        return

    files_to_process = glob.glob(LOCAL_DATA_PATH + SOURCE + '*.pkl')

    for file in files_to_process:

        print(f"\n Preprocessing {file} ..")

        df = get_pandas([file])

        if SOURCE=='raw/':
            df = expand_raw_cols(df)

        df = drop_cols(df)

        print(f"\n Engineering new features {file} ..")

        df = put_together(df)

        save_pandas(file,
                    'processed',
                    df)

    print(f"\n✅ Preprocessing: COMPLETE")

    return None

def train():

    print(f"\n Gathering processed files ..")

    # Retrieve processed files from processed directory
    processed_files = glob.glob(LOCAL_DATA_PATH + 'processed/*.pkl')

    # Concatenate all processed files
    df = pd.DataFrame()
    for file in processed_files:
        print(f"\n Loading file {file} ..")
        df = pd.concat([df, pd.read_pickle(file)])

    print(f"\n✅ Processed files: COMPLETE")

    df_train_KNN, df_test = train_test_split(df, train_size=0.7)

    # Train the KNN on the full training split as no validation required
    X_train_KNN = df_train_KNN.drop(['user_pseudo_id','target_variable'], axis=1).copy()

    # Split train into a train and validation split for classification modelling
    df_train_XGB, df_val_XGB = train_test_split(df_train_KNN, train_size=0.7)

    # Reset all indexes to zero
    df_train_KNN.reset_index(inplace=True, drop=True)
    df_train_XGB.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val_XGB.reset_index(inplace=True, drop=True)

    print(f"\n Saving test data ..")
    # Save test data to the test repositry to be used for model evaluation
    save_pandas('test_data',
                'test',
                df_test)

    # Flip the 0 & 1 in the target variable as we are targeting non-purchasers
    y_train_XGB = (df_train_XGB['target_variable'] == 0) * 1
    y_val_XGB = (df_val_XGB['target_variable'] == 0) * 1

    # Drop target from training data
    X_train_XGB = df_train_XGB.drop(['user_pseudo_id','target_variable'], axis=1).copy()
    X_val_XGB = df_val_XGB.drop(['user_pseudo_id','target_variable'], axis=1).copy()

    # Get preprocessing pipeline
    pipe, binary_col, robust_scaling_cat, standard_scaling_cat = pipeline_normalizer(X_train_XGB)

    # Preprocess training data
    X_train_XGB_preproc = pipe.fit_transform(X_train_XGB)
    X_train_KNN_preproc = pipe.transform(X_train_KNN)
    X_val_XGB_preproc = pipe.transform(X_val_XGB)

    # Create classification model
    XGBModel = None
    # Potentially retraining on seen data if we reload the
    # model here with the current structure
    # XGBModel = load_model('xgb_model')

    if XGBModel is None:
        XGBModel = XGBClassifier(learning_rate=0.1,
                                 max_depth = 10,
                                 n_estimators=500)

    print(f"\n Training model ..")

    # Train classification model
    XGBModel.fit(X_train_XGB_preproc,
                 y_train_XGB,
                 verbose=False,
                 eval_set=[(X_train_XGB_preproc, y_train_XGB),
                           (X_val_XGB_preproc, y_val_XGB)],
                 early_stopping_rounds=30)

    # Save classification model
    print(f"\n Saving XGB model ..")
    save_model('xgb_model',
               XGBModel)

    # Save fitted pipeline
    print(f"\n Saving pipeline ..")
    save_model('fitted_pipe',
                pipe)

    # Create KMeans model
    KModel = None
    # Potentially retraining on seen data if we reload the
    # model here with the current structure
    #KModel = load_model('k_model')

    if KModel is None:
        KModel = KMeans(n_clusters = 5, max_iter = 300)

    # Cluster on PCA
    pca = PCA(n_components=3, whiten=True)
    X_train_KNN_preproc_proj = pd.DataFrame(pca.fit_transform(X_train_KNN_preproc))

    # Train KMeans
    KModel.fit(X_train_KNN_preproc_proj)

    # Save KMeans model
    save_model('k_model',
               KModel)

    labelling = KModel.labels_

    test_labelled = pd.concat([df_train_KNN,pd.Series(labelling)],axis=1).rename(columns={0:"label"})
    test_labelled = test_labelled.set_index('user_pseudo_id')

    customer_mixes = {}

    for cluster in np.unique(labelling):
        customer_mixes[cluster] = test_labelled[test_labelled.label == cluster]

    breakpoint()

    return None

def evaluate():

    print(f"\n Loading models ..")

    XGBModel = load_model('xgb_model')
    KModel = load_model('k_model')
    pipe = load_model('fitted_pipe')

    print(f"\n Loading test data ..")
    processed_files = glob.glob(LOCAL_DATA_PATH + 'test/*.pkl')

    # Concatenate all processed files
    df_test = pd.DataFrame()
    for file in processed_files:
        print(f"\n Loading file {file} ..")
        df_test = pd.concat([df_test, pd.read_pickle(file)])

    y_test = (df_test['target_variable'] == 0) * 1
    X_test = df_test.drop(['user_pseudo_id', 'target_variable'], axis=1).copy()

    X_test_preproc = pipe.transform(X_test)

    print(f"\n Evaluating XGB ..")

    # Evaluate Laolu model on Test
    y_pred = XGBModel.predict(X_test_preproc)

    precision = precision_score(y_test, y_pred)

    print(f"\n XGB precision",precision)

    print(f"\n Evaluating KMeans ..")

    # Cluster on PCA
    pca = PCA(n_components=3, whiten=True)
    X_test_preproc_proj = pd.DataFrame(pca.fit_transform(X_test_preproc))

    KModel.predict(X_test_preproc_proj)

    return precision

def predict():

    print(f"\n Prediction ..")

    # Load Laolu model

    # Predict on Laolu model

if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    predict()
