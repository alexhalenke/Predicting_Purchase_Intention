from Predicting_Purchase_Intention.utils.params import LOCAL_DATA_PATH, SOURCE
from Predicting_Purchase_Intention.utils.clean_data import clean_data, drop_cols, expand_raw_cols
from Predicting_Purchase_Intention.utils.data import get_pandas, save_pandas
from Predicting_Purchase_Intention.utils.preprocess import put_together

import pandas as pd
import glob

def preprocess():

    if SOURCE=='processed':
        return

    files_to_process = glob.glob(LOCAL_DATA_PATH + SOURCE + '*.pkl')

    print(SOURCE)
    print([files_to_process[-1]])

    for file in [files_to_process[-1]]:

        print(f"\n Preprocessing {file} ..")

        df = get_pandas([file])

        if SOURCE=='raw':
            df = expand_raw_cols(df)

        df = drop_cols(df)

        print(f"\n Engineering new features {file} ..")

        df = put_together(df)

        save_pandas(file,
                    df)

    print(f"\n✅ Preprocessing: COMPLETE")

    return None

if __name__ == '__main__':
    preprocess()
    # Clean the data
    # Process the data
    # Run the model
