from Predicting_Purchase_Intention.utils.params import (LOCAL_DATA_PATH, SOURCE)

import pandas as pd

def get_pandas(files: list) -> pd.DataFrame:
    '''Retrieve source data'''

    df = pd.DataFrame()

    for file in files:
        df = pd.concat([df, pd.read_pickle(file)])

    return df


def save_pandas(file: str,
                stream: str,
                df: pd.DataFrame):

    file = file.split('/')[-1]

    if file[:-4] != '.pkl':
        file = file + '.pkl'

    file_name = f'{LOCAL_DATA_PATH}{stream}/{file}'

    print(f"\n Saving {file_name}")

    df.to_pickle(file_name)
