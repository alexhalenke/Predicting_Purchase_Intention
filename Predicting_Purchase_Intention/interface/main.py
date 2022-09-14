from Predicting_Purchase_Intention.utils.params import LOCAL_DATA_PATH, SOURCE
from Predicting_Purchase_Intention.utils.clean_data import clean_data, drop_cols, expand_raw_cols
from Predicting_Purchase_Intention.utils.data import get_pandas, save_pandas
from Predicting_Purchase_Intention.utils.preprocess import put_together
from Predicting_Purchase_Intention.utils.model import pipeline_normalizer
from Predicting_Purchase_Intention.utils.registry import save_model, load_model
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

    df_train, df_test = train_test_split(df, train_size=0.7)
    df_train, df_val = train_test_split(df_train, train_size=0.7)

    print(f"\n Saving test data ..")
    # Save test data to the test repositry to be used for model evaluation
    save_pandas('test_data',
                'test',
                df_test)

    # Flip the 0 & 1 in the target variable as we are targeting non-purchasers
    y_train = (df_train['target_variable'] == 0) * 1
    y_val = (df_val['target_variable'] == 0) * 1

    # Drop target from training data
    X_train = df_train.drop('target_variable', axis=1).copy()
    X_val = df_val.drop('target_variable', axis=1).copy()

    # ASK LALOU WHERE HE DROPS USER_PSEUDO_ID
    # I DON'T THINK WE NEED IT HERE?
    # Drop pseudo id for modelling, retain it for K Clustering
    X_train.drop('user_pseudo_id', axis=1, inplace=True)

    # Get preprocessing pipeline
    pipe, binary_col, robust_scaling_cat, standard_scaling_cat = pipeline_normalizer(X_train)

    # Preprocess training data
    X_train_preproc = pipe.fit_transform(X_train)
    X_val_preproc = pipe.transform(X_val)

    # Create Laolu model
    XGBModel = None
    XGBModel = load_model('xgb_model')

    if XGBModel is None:
        XGBModel = XGBClassifier(learning_rate=0.1,
                                 max_depth = 10,
                                 n_estimators=500)

    print(f"\n Training model ..")

    # Train Laolu model
    XGBModel.fit(X_train_preproc,
                 y_train,
                 verbose=False,
                 eval_set=[(X_train_preproc, y_train),
                           (X_val_preproc, y_val)],
                 early_stopping_rounds=30)

    print(f"\n Saving model ..")

    # Save Laolu model
    save_model('xgb_model',
               XGBModel)

    # Save fitted pipeline
    save_model('fitted_pipe',
                pipe)

    return None

def evaluate():

    print(f"\n Evaluating model ..")

    XGBModel = load_model('xgb_model')
    pipe = load_model('fitted_pipe')

    processed_files = glob.glob(LOCAL_DATA_PATH + 'test/*.pkl')

    # Concatenate all processed files
    df_test = pd.DataFrame()
    for file in processed_files:
        print(f"\n Loading file {file} ..")
        df_test = pd.concat([df_test, pd.read_pickle(file)])

    y_test = (df_test['target_variable'] == 0) * 1
    X_test_pseudo = df_test['user_pseudo_id'].copy()
    X_test = df_test.drop(['user_pseudo_id', 'target_variable'], axis=1).copy()

    X_test_preproc = pipe.transform(X_test)

    # Evaluate Laolu model on Test
    y_pred = XGBModel.predict(X_test_preproc)

    precision = precision_score(y_test, y_pred)

    print(f"\n Model precision",precision)

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
