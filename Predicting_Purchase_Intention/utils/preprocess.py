import pandas as pd
import numpy as np
import datetime
from Predicting_Purchase_Intention.utils.page_category import page_category
from sklearn.preprocessing import OneHotEncoder


'''
0. Run all 3 Functions
'''
def target_slicing(df:pd.DataFrame) -> pd.DataFrame:
    a1 = convert_datatypes_filter(df)
    a1 = input_target_variable(a1)
    b1 = slicing_purchasers(a1)
    return b1

'''
1. Convert the columns we are gonna work with into the correct data types
'''
def convert_datatypes_filter(df:pd.DataFrame) -> pd.DataFrame:
    index_drop = df.index[df['event_date']==1].tolist()
    index_drop.extend(df.index[df['user_first_touch_timestamp']=='<Other>'].tolist())
    df = df.drop(index=index_drop)
    for column in df:
        if 'event_date' in column:
            df[column] = pd.to_datetime(df[column],yearfirst=True, format='%Y%m%d')
        if 'timestamp' in column:
            df[column] = pd.to_datetime(df[column], unit='us')
        if 'user_pseudo_id' in column:
            df[column] = df[column].astype(str)
    return df

'''
2. Input Data Variable
'''
def input_target_variable(df:pd.DataFrame) -> pd.DataFrame:
    #add target_variable column for visitors with at least 1 purchase
    purchasers = list(df[df['event_name']=='purchase']['user_pseudo_id'].unique())
    df.loc[df['user_pseudo_id'].isin(purchasers), 'target_variable'] = 1
    df['target_variable'] = df.target_variable.fillna(0)
    return df

'''
3. Slicing
'''
def slicing_purchasers(df:pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(by=['event_timestamp'], ascending=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'],inplace=True)
    purchasers = df[df['event_name']=='purchase']['user_pseudo_id'].unique().tolist()

    drop_rows = []
    for purchaser in purchasers:

        # List of unique purchasers
        df_purchaser = df[df['user_pseudo_id'] == purchaser].copy()

        # Index of first purchase (sorted by time)
        first_purch_idx = df_purchaser[df_purchaser['event_name'] == 'purchase'].index[0]

        # Index of all purchasers events
        full_purch_idx = df_purchaser.index.to_list()

        # Gather index of all events beyond and including first purchase
        for idx in full_purch_idx:
            if idx >= first_purch_idx:
                drop_rows.append(idx)

    # Drop all events beyond and including first purchase
    df.drop(drop_rows, inplace=True)

    # #new User ID based on repeat purchases
    # df['iter_purch_tf'] = np.where(df['event_name'] == 'purchase', 1.0, 0.0)
    # df['user_pseudo_id'] = df['user_pseudo_id'].astype(str)
    # df = df.sort_values(by=['event_timestamp'], ascending=True)
    # df['num_previous_purchase'] = df.groupby('user_pseudo_id')['iter_purch_tf'].transform(lambda x: x.cumsum().shift())
    # df['num_previous_purchase'] = (df['num_previous_purchase'].fillna(0)).astype(int)
    # df['new_user_id'] = df['user_pseudo_id'].astype(str) + df['num_previous_purchase'].astype(str)
    # df['user_pseudo_id'] = df['new_user_id']
    # df = df.drop(columns = ['new_user_id'])
    # #Create DF with amount purchases per User Pseudo ID
    # all_purchases = df[df['event_name']=='purchase']
    # all_purchases2 = all_purchases.copy()
    # all_purchases2['amount_purchases'] = 1
    # df_amount_purchases = pd.pivot_table(data=all_purchases2, index='user_pseudo_id', values='event_name', aggfunc='count').reset_index().rename(columns={'event_name': 'amount_purchases'})
    # #Merge with main DF
    # df1 = df.merge(df_amount_purchases,on='user_pseudo_id', how='outer').fillna(0)
    # #New column with timestamp of first purchase
    # all_purchases_timestamp = all_purchases[['user_pseudo_id','event_timestamp']]
    # first_purchase = pd.DataFrame(all_purchases_timestamp.groupby(['user_pseudo_id'])['event_timestamp'].agg('min').reset_index().rename(columns={'event_timestamp': 'date_first_purchase'}))
    # #Merge with main DF
    # df2 = df1.merge(first_purchase, on='user_pseudo_id', how='outer')
    # df2.reset_index(inplace=True, drop=True)
    # df2['event_params_ga_session_number'] = df2['event_params_ga_session_number'].astype(float)
    # df2.drop(columns=['iter_purch_tf', 'amount_purchases', 'date_first_purchase'], inplace=True)


    return df


'''
0. Run all 2 Functions
'''
def create_features(df:pd.DataFrame) -> pd.DataFrame:
    a1 = creation_own_features(df)
    a2 = event_pagenames(df)
    b1 = a1.merge(a2, how='left',on='user_pseudo_id')
    b2 = b1.fillna(0)
    return b1

'''
1. Creation of own features
'''
def creation_own_features(df:pd.DataFrame) -> pd.DataFrame:
    #visits per user_pseudo_id
    df_visits = df.groupby('user_pseudo_id').agg({'event_params_ga_session_number':'max'}).reset_index().rename(columns={'event_params_ga_session_number': 'visits_per_user_pseudo_id'})
    #pageviews per user_pseudo_id
    df_pageviews_filter = df[df['event_name']=='page_view'][['user_pseudo_id', 'event_name','event_date' ]]
    df_pageviews = pd.pivot_table(data=df_pageviews_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'pageviews_per_user_pseudo_id'})
    #events per user_pseudo_id
    df_events = df.drop(df.index[df['event_name'] == 'purchase'])
    df_events = df.groupby('user_pseudo_id').agg({'event_params_ga_session_number':'count'}).reset_index().rename(columns={'event_params_ga_session_number': 'events_per_visitor'})
    df_events['events_per_visitor'] = df_events['events_per_visitor'].astype(float)
    #engagement time per user_pseudo_id
    df_engagement_time = df.groupby('user_pseudo_id').agg({'event_params_engagement_time_msec':'sum'}).reset_index().rename(columns={'event_params_engagement_time_msec': 'engagement_time_per_visitor'})
    #clicks per user_pseudo_id
    df_clicks_filter = df[df['event_name']=='click'][['user_pseudo_id', 'event_name','event_date' ]]
    df_clicks = pd.pivot_table(data=df_clicks_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'clicks_per_visitor'})
    #engagement per user_pseudo_id
    df_user_engagement_filter = df[df['event_name']=='user_engagement'][['user_pseudo_id', 'event_name','event_date' ]]
    df_user_engagements = pd.pivot_table(data=df_user_engagement_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'user_engagements_per_visitor'})
    #scrollings per user_pseudo_id
    df_scrollings_filter = df[df['event_name']=='scroll'][['user_pseudo_id', 'event_name','event_date' ]]
    df_scrollings = pd.pivot_table(data=df_scrollings_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'scrolls_per_visitor'})
    #Merge all of it
    df_level_user_pseudo_id = df_visits.merge(df_pageviews,on='user_pseudo_id', how='left').merge(df_events,on='user_pseudo_id', how='left').merge(df_engagement_time,on='user_pseudo_id', how='left').merge(df_clicks,on='user_pseudo_id', how='left').merge(df_user_engagements,on='user_pseudo_id', how='left').merge(df_scrollings,on='user_pseudo_id', how='left')
    return df_level_user_pseudo_id

'''
2. Create all combinatins of event_pagenames
'''
def event_pagenames(df:pd.DataFrame) -> pd.DataFrame:
    #Categorize Page Titles
    df = page_category(df)
    #Create column with unique combinations
    df['unique_combinations'] = df['event_name']+ ' & ' +df['Page Category']
    df = df[['user_pseudo_id','unique_combinations', 'Page Category']]
    df = pd.pivot_table(df, values='Page Category', index='user_pseudo_id', columns='unique_combinations',aggfunc='count')
    df = df.reset_index()
    return df

'''
0. Run all 2 Functions
'''
def categorise_all(df:pd.DataFrame) -> pd.DataFrame:
    x1 = get_categorical_features(df)
    x2 = one_hot_encode(x1)
    return x2

'''
1. Get categorical Features
'''
def get_categorical_features(df:pd.DataFrame) -> pd.DataFrame:
    categorical_features = ['user_pseudo_id',
                            'event_params_link_domain',
                            'device_category',
                            'device_operating_system',
                            'device_device_web_info_browser',
                            'geo_country',
                            'traffic_source_medium']
    df_categorical = df[categorical_features].copy()
    df_categorical['outbound_click'] = (df_categorical['event_params_link_domain'].isna() == False) * 1
    df_categorical.drop('event_params_link_domain', axis=1, inplace=True)
    return df_categorical

'''
2. One Hot Encode Them
'''
def one_hot_encode(df:pd.DataFrame) -> pd.DataFrame:
    df['device_device_web_info_browser'] = series_other(df['device_device_web_info_browser'], 2, 'web_info_browser_other')
    df['geo_country'] = series_other(df['geo_country'], 5, 'geo_country_other')
    df_ohe = df.drop('outbound_click', axis=1)
    df_ohe.reset_index(inplace=True, drop=True)
    df_ohe['user_pseudo_id'] = df_ohe['user_pseudo_id'].astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float64)
    df_ohe_transformed = ohe.fit_transform(df_ohe[['device_category','device_operating_system','device_device_web_info_browser','geo_country','traffic_source_medium']])
    df_ohe_transformed_df = pd.DataFrame(df_ohe_transformed, columns=ohe.get_feature_names_out())
    df_final_OHE = pd.merge(df_ohe_transformed_df, df_ohe[['user_pseudo_id']], how='left', left_index=True, right_index=True)
    df_final_OHE = df_final_OHE.groupby(by='user_pseudo_id').mean().round(decimals=2)
    return df_final_OHE

'''
3. Help for One Hot Encoding
'''
def series_other(series: pd.Series,
                 cut_off: int,
                 rename: str='Other') -> pd.Series:
    series = series.map(
        {**dict(zip(series.value_counts().index.tolist()[:cut_off],
                    series.value_counts().index.tolist()[:cut_off])),
        **dict.fromkeys(series.value_counts().index.tolist()[cut_off:],rename)}
    )
    return series

'''
All the preprocessing for creating a DataFrame
'''
def put_together(df:pd.DataFrame) -> pd.DataFrame:


    a1 = target_slicing(df)
    d1 = a1[['user_pseudo_id','target_variable']]
    d1 = d1.drop_duplicates()
    b1 = create_features(a1)
    b2 = categorise_all(a1).reset_index()
    c1 = b1.merge(b2, how='left',on='user_pseudo_id').merge(d1, how='left',on='user_pseudo_id').fillna(0)


    # Drop highly correlated features
    columns = c1.columns
    event_cols = [column for column in columns if '&' in column]
    event_cols.append('target_variable')
    df_events = c1[event_cols].copy()
    #df_events = df_events.fillna(0)
    df_events = df_events.clip(upper=1)
    correlations = abs(df_events.corr().fillna(1)['target_variable']).sort_values(ascending=False)
    drop_cols = correlations[correlations > 0.85].index.to_list()
    drop_cols.remove('target_variable')
    c1=c1.drop(columns=drop_cols)
    c1.drop('engagement_time_per_visitor', axis=1, inplace=True)

    columns = c1.columns
    drop_cols1 = [col for col in columns if 'add_payment' in col]
    drop_cols2 = [col for col in columns if 'add_shipping' in col]
    drop_cols = drop_cols1 + drop_cols2
    c1.drop(drop_cols, axis=1, inplace=True)

    #breakpoint()

    return c1
