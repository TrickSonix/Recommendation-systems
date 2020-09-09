import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000):

    items_price = data.groupby('item_id')[['quantity', 'sales_value']].sum().reset_index()
    items_price['price'] = items_price['sales_value']/items_price['quantity']
    filtered_items = items_price.loc[items_price['price']>1]
    filtered_items = filtered_items.loc[filtered_items['price']<30]
    item_list = filtered_items['item_id'].unique().tolist()

    user_count = data.groupby('item_id')['user_id'].nunique().reset_index().rename(columns={'user_id':'user_count'})
    user_count['known'] = user_count['user_count']/data['user_id'].nunique()
    user_count = user_count.loc[user_count['known'] > 0.5]
    item_list = [x for x in item_list if x not in user_count['item_id'].unique().tolist()]
    filtered_items = filtered_items.loc[filtered_items['item_id'].isin(item_list)]

    top_n = filtered_items.sort_values('quantity', ascending=False).head(take_n_popular)['item_id'].tolist()

    data.loc[~data['item_id'].isin(top_n), 'item_id'] = 999999
    return data