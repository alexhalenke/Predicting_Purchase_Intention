import pandas as pd
import numpy as np

def expand_raw_cols(series: pd.Series,
                    verbose: bool=True) -> pd.DataFrame:
    '''Expands packed features into additional columns'''

    if verbose and series.name % 5000 == 0:
      print(series.name, ' rows expanded ..')

    for dicts in series['event_params']:
        if dicts['key'] == 'page_title':
            series['event_params_page_title'] = dicts['value']['string_value']
        elif dicts['key'] == 'link_domain':
            series['event_params_link_domain'] = dicts['value']['string_value']
        elif dicts['key'] == 'engagement_time_msec':
            series['event_params_engagement_time_msec'] = dicts['value']['int_value']
        elif dicts['key'] == 'ga_session_id':
            series['event_params_ga_session_id'] = dicts['value']['int_value']
        elif dicts['key'] == 'ga_session_number':
            series['event_params_ga_session_number'] = dicts['value']['int_value']

    series['user_ltv_revenue'] = series['user_ltv']['revenue']
    series['device_category'] = series['device']['category']
    series['device_operating_system'] = series['device']['operating_system']
    series['device_device_web_info_browser'] = series['device']['web_info']['browser']
    series['geo_country'] = series['geo']['country']
    series['traffic_source_medium'] = series['traffic_source']['medium']
    series['ecommerce_total_item_quantity'] = series['ecommerce']['total_item_quantity']
    series['ecommerce_unique_items'] = series['ecommerce']['unique_items']

    return series

def drop_cols(df:pd.DataFrame) -> pd.DataFrame:
    '''Drop default columns from data'''

    fields_to_drop = ['event_params',
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
                      'event_params_payment_type',
                      'event_value_in_usd',
                      'event_bundle_sequence_id',
                      'event_params_page_location',
                      'device_operating_system_version',
                      'device_web_info_browser_version',
                      'traffic_source_name',
                      'ecommerce_purchase_revenue_in_usd',
                      'ecommerce_purchase_revenue',
                      'ecommerce_tax_value_in_usd',
                      'ecommerce_tax_value',
                      'event_params_campaign',
                      'event_params_page_referrer',
                      'event_params_link_url',
                      'event_params_transaction_id',
                      'event_params_tax']

    for field in fields_to_drop:
        if field in df.columns:
            df.drop(field, axis=1, inplace=True)

    # Remove any features that are null throughout the dataset
    df_nans = df.isnull().sum() / len(df)
    df_nans = df_nans[df_nans == 1]
    nans_to_drop = df_nans.index.tolist()
    df.drop(nans_to_drop, axis=1, inplace=True)

    return df

def clean_data(df:pd.DataFrame) -> pd.DataFrame:

    print('Expanding raw columns ..')

    expansion_cols = ['event_params',
                      'user_ltv',
                      'device', 'geo',
                      'traffic_source',
                      'ecommerce']

    df_expanded = df[expansion_cols
                    ].apply(expand_raw_cols,
                            axis=1,
                            ).drop(columns=expansion_cols)

    df['place_holder'] = np.arange(0, len(df), 1)
    df_expanded['place_holder'] = np.arange(0, len(df), 1)
    df = df.merge(df_expanded, how='left', on='place_holder')
    df = df.drop(columns=['place_holder'])

    print('Dropping unecessary columns ..')
    df = drop_cols(df)

    return df
