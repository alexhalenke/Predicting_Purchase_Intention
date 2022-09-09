import pandas as pd
import numpy as np
import datetime
import glob
from Predicting_Purchase_Intention.utils.clean_data import clean_data
from Predicting_Purchase_Intention.utils.page_category import page_category


def input_target_variable(df:pd.DataFrame) -> pd.DataFrame:
    #add target_variable column
    df['target_variable'] = None
    #create a list with all purchasers
    purchasers = list(df[df['event_name']=='purchase']['user_pseudo_id'])
    for i in purchasers:
        for index, row in df[df['user_pseudo_id']==i].iterrows():
            df['target_variable'].iloc[index]=1
     #Fill the NaNs with 0s
    df['target_variable'] = df.target_variable.fillna(0)
    #Clean target_variable column
    list_y = list(set(df['target_variable'].values))
    list_y = list_y.remove(0)
    df['target_variable'] = df['target_variable'].replace(to_replace=list_y, value = 1)
    return df


def delete_transactions_after_purchase(df:pd.DataFrame) -> pd.DataFrame:
    #Make the timestamp a timestamp
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], unit='us')
    #Create a new DF with the timestamp of purchase
    last_event = df[df['event_name']=='purchase'][['user_pseudo_id','event_timestamp']]
    last_event.rename(columns={'event_timestamp':'event_timestamp_purchase'},inplace=True)
    #Merge last_event with df
    df = df.merge(last_event, on='user_pseudo_id', how='left')
    #Drop all rows with 'event_timestamp' > 'event_timestamp_purchase'
    df = df.drop(df[df.event_timestamp > df.event_timestamp_purchase].index)
    return df


def creation_dataframe_and_own_features_(df:pd.DataFrame) -> pd.DataFrame:
    #visits per user_pseudo_id
    df_visits = df.groupby('user_pseudo_id').agg({'event_params_ga_session_number':'max'}).reset_index().rename(columns={'event_params_ga_session_number': 'visits_per_user_pseudo_id'})
    #pageviews per user_pseudo_id
    df_pageviews_filter = df[df['event_name']=='page_view'][['user_pseudo_id', 'event_name','event_date' ]]
    df_pageviews = pd.pivot_table(data=df_pageviews_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'pageviews_per_user_pseudo_id'})
    #events per user_pseudo_id
    df_events = df.groupby('user_pseudo_id').agg({'event_params_ga_session_number':'count'}).reset_index().rename(columns={'event_params_ga_session_number': 'events_per_visitor'})
    #engagement time per user_pseudo_id
    df_engagement_time = df.groupby('user_pseudo_id').agg({'event_params_engagement_time_msec':'sum'}).reset_index().rename(columns={'event_params_engagement_time_msec': 'engagement_time_per_visitor'})
    #clicks per user_pseudo_id
    df_clicks_filter = df[df['event_name']=='click'][['user_pseudo_id', 'event_name','event_date' ]]
    df_clicks = pd.pivot_table(data=df_clicks, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'clicks_per_visitor'})
    #engagement per user_pseudo_id
    df_user_engagement_filter = df[df['event_name']=='user_engagement'][['user_pseudo_id', 'event_name','event_date' ]]
    df_user_engagements = pd.pivot_table(data=df_user_engagement_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'user_engagements_per_visitor'})
    #scrollings per user_pseudo_id
    df_scrollings_filter = df[df['event_name']=='scroll'][['user_pseudo_id', 'event_name','event_date' ]]
    df_scrollings = pd.pivot_table(data=df_scrollings_filter, index='user_pseudo_id', columns='event_name', aggfunc='count').droplevel(level=1, axis=1).reset_index().rename(columns={'event_date': 'scrolls_per_visitor'})

    #Merge all of it
    df_level_user_pseudo_id = df_visits.merge(df_pageviews,on='user_pseudo_id', how='outer').merge(df_events,on='user_pseudo_id', how='outer').merge(df_engagement_time,on='user_pseudo_id', how='outer').merge(df_clicks,on='user_pseudo_id', how='outer').merge(df_user_engagements,on='user_pseudo_id', how='outer').merge(df_scrollings,on='user_pseudo_id', how='outer')
    return df_level_user_pseudo_id


def features_part1_event_pagenames(df:pd.DataFrame) -> pd.DataFrame:
    #Categorize Page Titles
    df = page_category(df)
    #Create column with unique combinations
    df['unique_combinations'] = df['event_name']+ ' & ' +df['Page Category']
    df = df.set_index('user_pseudo_id')
    df = df[['user_pseudo_id','unique_combinations', 'Page Category']]
    df = pd.pivot_table(df, values='Page Category', index='user_pseudo_id', columns='unique_combinations',aggfunc='count')
    return df


def feature_engineering(df:pd.DataFrame):
    df_merge = df_level_user_pseudo_id.merge(df, how='outer',on='user_pseudo_id')
    return df_merge
