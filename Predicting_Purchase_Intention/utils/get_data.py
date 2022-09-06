import pandas as pd
from google.cloud import bigquery

def get_raw_data(project=None,
                 query="""
                       SELECT *
                       FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
                       LIMIT 1
                       """,
                 drop=False):

    if project == None:
        return "Please input big query project ID"

    client = bigquery.Client(project=project)

    results = client.query(query)
    df = pd.DataFrame(results.result().to_dataframe())

    expanded_fields = []

    for index, row in df.iterrows():
        '''Retrieve and expand the dataset'''

        for col, field in enumerate(row):

            if type(field) == list:
                if df.columns[col] not in expanded_fields:
                    expanded_fields.append(df.columns[col])

                if df.columns[col] == 'event_params':
                    for dicts in field:
                        for data_type in dicts['value'].keys():
                            if dicts['value'][data_type] != None:
                                df.loc[index,f'{df.columns[col]}_{dicts["key"]}'] = dicts['value'][data_type]
                                break

            if type(field) == dict:
                if df.columns[col] not in expanded_fields:
                    expanded_fields.append(df.columns[col])

                for key in field.keys():
                    if key != 'web_info':
                        df.loc[index,f'{df.columns[col]}_{key}'] = field[key]
                    else:
                        df.loc[index,'device_web_info_browser'] = field[key]['browser']
                        df.loc[index,'device_web_info_browser_version'] = field[key]['browser_version']

        if drop:
            df.drop(expanded_fields, axis=1, inplace=True)

    return df
