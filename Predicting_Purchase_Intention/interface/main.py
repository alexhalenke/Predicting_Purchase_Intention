from Predicting_Purchase_Intention.utils.params import LOCAL_DATA_PATH, SOURCE
from Predicting_Purchase_Intention.utils.clean_data import clean_data, drop_cols, expand_raw_cols
from Predicting_Purchase_Intention.utils.data import get_pandas, save_pandas
from Predicting_Purchase_Intention.utils.preprocess import put_together

import pandas as pd
import glob

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
                    df)

    print(f"\n✅ Preprocessing: COMPLETE")


    return None

def train():

    print(f"\n Gathering processed files ..")

    processed_files = glob.glob(LOCAL_DATA_PATH + 'processed/*.pkl')

    df = pd.DataFrame()
    for file in processed_files:
        print(f"\n Loading file {file} ..")
        df = pd.concat([df, pd.read_pickle(file)])

    print(f"\n✅ Processed files: COMPLETE")

    # Create Laolu model

    # Train Laolu model

    # Save Laolu model

    return None

def evaluate():

    print(f"\n Evaluating model ..")

    # Load Laolu model

    # Evaluate Laolu model

def predict():

    print(f"\n Prediction ..")

    # Load Laolu model

    # Predict on Laolu model

if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    predict()
