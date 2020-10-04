import pandas as pd
import numpy as np
import pickle
import multiprocessing

from catboost import CatBoostClassifier

from src.metrics import money_precision_at_k
from src.recommenders import MainRecommender
from src.utils import prefilter_items, postfilter_items, DataTransformer
from src.settings import MAIN_DATA_PATH, ITEM_FEATURES_PATH, USER_FEATURES_PATH, VAL_LVL_1_SIZE_WEEKS, VAL_LVL_2_SIZE_WEEKS, \
                         TEST_DATA_PATH, MODEL_LVL_1_PATH, MODEL_LVL_2_PATH


def get_recommendation_lvl_1(data, model):
    result = data.groupby('user_id')['item_id'].unique().reset_index()
    result.columns = ['user_id', 'actual']
    result['lvl_1_recs'] = result['user_id'].apply(lambda x: model.get_main_model_recommendations(x, N=6000))

    return result

def get_recommendation_lvl_2(data, model):

    predictions = model.predict_proba(data)
    data_with_preds = data.copy()
    data_with_preds['predictions'] = predictions[:, 1]
    user_id_list = []
    recs = []
    sorted_data = data_with_preds.sort_values(['user_id', 'predictions'], ascending=False)
    for user_id in data_with_preds['user_id'].unique():
        user_id_list.append(user_id)
        recs.append(sorted_data[sorted_data['user_id']==user_id]['item_id'].tolist())
        
    result = pd.DataFrame({'user_id':user_id_list, 'catboost_recs': recs})
    return result

def get_recommendation_df(data, model_lvl_1, model_lvl_2, items_sub_comm, prices, item_features, user_features):
    trans = DataTransformer()
    print('Construct level 1 recommendations DataFrame...')
    result = get_recommendation_lvl_1(data, model_lvl_1)
    result = postfilter_items(result, 'lvl_1_recs', items_sub_comm, prices, N=200)
    print('Construct level 2 recommendations DataFrame')
    result.rename(columns={'postfilter_lvl_1_recs': 'recommendations'}, inplace=True)
    X_test = trans.fit_transform(result, data, item_features, user_features)
    catboost_recs = get_recommendation_lvl_2(X_test, model_lvl_2)
    result = result.merge(catboost_recs, on='user_id', how='inner')
    result = postfilter_items(result, 'catboost_recs', items_sub_comm, prices, N=5)
    result['recommend_prices'] = result['postfilter_catboost_recs'].apply(lambda x: [prices[item] for item in x])

    return result

def get_results(main_data_path, item_features_path, user_features_path, val_lvl_1_size_weeks, val_lvl_2_size_weeks,
                model_lvl_1_path, model_lvl_2_path, test_data_path):
    
    print('Reading data...')
    data = pd.read_csv(main_data_path)
    item_features = pd.read_csv(item_features_path)
    user_features = pd.read_csv(user_features_path)
    item_features.columns = [col.lower() for col in item_features.columns]
    user_features.columns = [col.lower() for col in user_features.columns]

    item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
    user_features.rename(columns={'household_key': 'user_id'}, inplace=True)

    data_train_lvl_1 = data[data['week_no'] < data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)]
    data_val_lvl_1 = data[(data['week_no'] >= data['week_no'].max() - (val_lvl_1_size_weeks + val_lvl_2_size_weeks)) &
                        (data['week_no'] < data['week_no'].max() - (val_lvl_2_size_weeks))]

    data_train_lvl_2 = data_val_lvl_1.copy()
    data_val_lvl_2 = data[data['week_no'] >= data['week_no'].max() - val_lvl_2_size_weeks]
    
    prices = data.groupby('item_id')[['sales_value', 'quantity']].sum().reset_index()
    prices['price'] = prices['sales_value']/prices['quantity']
    prices.replace(np.inf, 0, inplace=True)
    prices = dict(zip(prices['item_id'], prices['price']))

    items_sub_comm = dict(zip(item_features['item_id'], item_features['sub_commodity_desc']))
    trans = DataTransformer()

    if test_data_path:
        test_data = pd.read_csv(test_data_path)

    if model_lvl_1_path and model_lvl_2_path:
        print('Reading models...')
        with open(model_lvl_1_path, 'rb') as f:
            model_lvl_1 = pickle.load(f)
        with open(model_lvl_2_path, 'rb') as f:
            model_lvl_2 = pickle.load(f)
        if test_data_path:
            result = get_recommendation_df(test_data, model_lvl_1, model_lvl_2, items_sub_comm, prices, item_features, user_features)           
        else:
            result = get_recommendation_df(data_val_lvl_2, model_lvl_1, model_lvl_2, items_sub_comm, prices, item_features, user_features)
    else:
        print('Prepare data for level 1 model fit...')
        data_train_lvl_1 = prefilter_items(data_train_lvl_1, 6000)
        print('Level 1 model fit...')
        num_threads = multiprocessing.cpu_count()//2 + 1
        model_lvl_1 = MainRecommender(data_train_lvl_1, weighting=None, n_factors=100, regularization=0.01, iterations=100, num_threads=num_threads)
        print('Construct level 1 recommendations DataFrame...')
        result_lvl_1 = get_recommendation_lvl_1(data_val_lvl_1, model_lvl_1)
        result_lvl_1 = postfilter_items(result_lvl_1, 'lvl_1_recs', items_sub_comm, prices, N=200)
        result_lvl_1.rename(columns={'postfilter_lvl_1_recs': 'recommendations'}, inplace=True)
        print('Prepare data for level 2 model fit...')
        data_train_lvl_2 = trans.fit_transform(result_lvl_1, data_val_lvl_1, item_features, user_features, with_targets=True)
        y = data_train_lvl_2['target']
        X = data_train_lvl_2.drop('target', axis=1)
        cat_features = ['department', 'brand', 'commodity_desc', 'sub_commodity_desc', 'curr_size_of_product',
                'age_desc', 'marital_status_code', 'income_desc', 'homeowner_desc', 'hh_comp_desc', 'household_size_desc', 
                'kid_category_desc']
        class_1_weight = len(y[y==0])/len(y[y==1])
        model_lvl_2 = CatBoostClassifier(n_estimators=300, max_depth=7, class_weights=[1, class_1_weight], cat_features=cat_features)
        print('Level 2 model fit...')
        model_lvl_2.fit(X, y)
        if test_data_path:
            result = get_recommendation_df(test_data, model_lvl_1, model_lvl_2, items_sub_comm, prices, item_features, user_features)           
        else:
            result = get_recommendation_df(data_val_lvl_2, model_lvl_1, model_lvl_2, items_sub_comm, prices, item_features, user_features)

    print('Calculating final metric...')
    money_precision_at_5 = result.apply(lambda row: money_precision_at_k(row['postfilter_catboost_recs'], 
                                                                            row['actual'], row['recommend_prices']), axis=1).mean()
    print(f'Money precision@5 for final recommendations:{money_precision_at_5}')
    result.to_csv('Final_recommendations.csv')
    return result


if __name__ == "__main__":
    get_results(MAIN_DATA_PATH, ITEM_FEATURES_PATH, USER_FEATURES_PATH, VAL_LVL_1_SIZE_WEEKS, VAL_LVL_2_SIZE_WEEKS,
                 MODEL_LVL_1_PATH, MODEL_LVL_2_PATH, TEST_DATA_PATH)
