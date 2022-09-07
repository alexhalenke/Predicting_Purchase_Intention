import pandas as pd

def drop_cols(df:pd.DataFrame) -> pd.DataFrame:
    '''Drop default columns from data'''

    fields_to_drop = [# May have been created on csv import
                     'Unnamed: 0',

                     # Fields have been unpacked into additional features
                     'event_params',
                     'privacy_info',
                     'user_properties',
                     'user_ltv',
                     'device',
                     'geo',
                     'traffic_source',
                     'ecommerce',
                     'items',

                     # Additional columns to be explicitly dropped
                     'stream_id',
                     'platform',
                     'event_params_clean_event',
                     'event_params_debug_mode',
                     'event_params_entrances',
                     'privacy_info_uses_transient_token',
                     'user_ltv_currency',
                     'device_mobile_marketing_name',
                     'device_is_limited_ad_tracking',
                     'geo_metro',
                     'ecommerce_transaction_id',
                     'event_params_term',
                     'event_params_percent_scrolled',
                     'event_params_search_term',
                     'event_params_unique_search_term',
                     'event_params_currency',
                     'event_params_payment_type']

    for field in fields_to_drop:
        print(field)
        if field in df.columns:
            df.drop(field, axis=1, inplace=True)

    # Remove any features that are null throughout the dataset
    df_nans = df.isnull().sum() / len(df)
    df_nans = df_nans[df_nans == 1]
    nans_to_drop = df_nans.index.tolist()
    df.drop(nans_to_drop, axis=1, inplace=True)

    return df


def clean_data(df:pd.DataFrame) -> pd.DataFrame:

    df = drop_cols(df)

    return df
