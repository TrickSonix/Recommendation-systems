import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000):
    """Фильтрация item_id происходимт по цене (>1 и <30), по известности пользователями, и top-n популярных"""

    items_price = data.groupby('item_id')[['quantity', 'sales_value']].sum().reset_index()
    items_price['price'] = items_price['sales_value']/items_price['quantity']
    items_price.replace(np.inf, 0, inplace=True)
    filtered_items = items_price.loc[items_price['price']>1]
    filtered_items = filtered_items.loc[filtered_items['price']<30]
    item_list = filtered_items['item_id'].unique().tolist()

    user_count = data.groupby('item_id')['user_id'].nunique().reset_index().rename(columns={'user_id':'user_count'})
    user_count['known'] = user_count['user_count']/data['user_id'].nunique()
    user_count = user_count.loc[user_count['known'] > 0.5]
    item_list = [x for x in item_list if x not in user_count['item_id'].unique().tolist()]
    filtered_items = filtered_items.loc[filtered_items['item_id'].isin(item_list)] 

    top_n = filtered_items.sort_values('sales_value', ascending=False).head(take_n_popular)['item_id'].tolist()

    result = data.copy()
    result.loc[~result['item_id'].isin(top_n), 'item_id'] = 999999
    return result

def postfilter_items(recommend_df, column_to_filter, items_sub_coms, item_prices_dict, N=5):
    """Фильтрация рекоммендаций модели в соответствии с бизнес-требованиями: 
    -Все товары из разных категорий
    -1 дорогой товар, > 7 долларов
    -2 новых товара (юзер никогда не покупал)"""

    recommend_list = recommend_df[column_to_filter].tolist()
    bought_list = recommend_df['actual'].tolist()
    result = []
    for recs, bought in zip(recommend_list, bought_list):
        user_recs = []
        expensive_item = False
        new_item1 = False
        new_item2 = False
        expensive_item_2 = False
        used_sub_commodities = []
        for rec in recs:
            if len(user_recs) < N-4:
                if items_sub_coms[rec] not in used_sub_commodities:
                    user_recs.append(rec)
                    used_sub_commodities.append(items_sub_coms[rec])
                    if item_prices_dict[rec] > 7:
                        expensive_item = True
                    if item_prices_dict[rec] >= 5:
                        expensive_item_2 = True
                    if rec not in bought:
                        if not new_item1:
                            new_item1 = True
                        else:
                            new_item2 = True
            else:
                if len(user_recs) == N:
                    break
                if items_sub_coms[rec] not in used_sub_commodities:
                    if rec not in bought and (not new_item1 or not new_item2):
                        user_recs.append(rec)
                        used_sub_commodities.append(items_sub_coms[rec])
                        if not new_item1:
                            new_item1 = True
                        else:
                            new_item2 = True
                        if item_prices_dict[rec] > 7:
                            expensive_item = True
                        if item_prices_dict[rec] >= 5:
                            expensive_item_2 = True
                    elif not expensive_item and item_prices_dict[rec] > 7:
                        user_recs.append(rec)
                        used_sub_commodities.append(items_sub_coms[rec])
                        expensive_item = True
                    elif not expensive_item_2 and item_prices_dict[rec] >= 5:
                        user_recs.append(rec)
                        used_sub_commodities.append(items_sub_coms[rec])
                        expensive_item_2 = True
                    else:
                        if new_item1 and new_item2 and expensive_item and expensive_item_2:
                            user_recs.append(rec)
                            used_sub_commodities.append(items_sub_coms[rec])
        
        assert len(user_recs) == N, f'Количество рекомендаций меньше {N}'

        result.append(user_recs)

    result_df = recommend_df.copy()
    result_df[f'postfilter_{column_to_filter}'] = result

    return result_df


class DataTransformer:
    """Класс для создания датасета для обучения модели второго уровня.
    Датасет создается из четырех DataFrame'ов:
    -DataFrame с рекомендациями товаров для каждого юзера. (Колонки 'user_id' и 'recommendations') (recommendations_df)
    -DataFrame покупок (purchase_df)
    -Item Features DataFrame
    -User Features DataFrame
    Новые фичи после преобразования:
        Фичи user_id:
        -Средний чек
        -Средняя сумма покупки 1 товара в каждой sub_commodity_desc
        -Средняя сумма покупки 1 товара в каждой commodity_desc
        -Средняя сумма покупки 1 товара в каждом department
        -Количество покупок в каждой sub_commodity_desc
        -Количество покупок в каждой commodity_desc
        -Количество покупок в каждом department
        -Частотность покупок раз/месяц
        -Доля покупок в выходные (кол-во айтемов в выходные/общее кол-во приобретенных айтемов)
        -Средняя сумма покупок/месяц
        -Среднее количество(quantity) покупаемых айтемов за одну покупку
        Фичи item_id:
        -Кол-во покупок в неделю
        -Среднее кол-во покупок 1 товара в sub_commodity_desc в неделю
        -Среднее кол-во покупок 1 товара в commodity_desc в неделю
        -Цена товара
        -Средняя цена товара в sub_commodity_desc
        -Цена/Средняя цена товара в sub_commodity_desc
        Фичи user_id-item_id:
        -Средняя сумма покупки 1 товара в каждой sub_commodity_desc - Цена товара
        -Средняя сумма покупки 1 товара в каждой commodity_desc - Цена товара
        -Средняя сумма покупки 1 товара в каждом department - Цена товара
        -Количество покупок в каждом department конкретного юзера в неделю - Среднее кол-во покупок всеми юзерами в department в неделю"""
    def __init__(self):
        pass

    def _mean_check(self, purchase_df):
        """DataFrame для фичи Средний чек"""

        mean_check_by_user = purchase_df.groupby(['user_id', 'basket_id'])['sales_value'].sum().reset_index().groupby('user_id')['sales_value'].mean().reset_index()
        mean_check_by_user.rename(columns={'sales_value': 'mean_check'}, inplace=True)

        return mean_check_by_user

    def _mean_sum_purchase_sub_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Средняя сумма покупки 1 товара в каждой sub_commodity_desc"""

        data = purchase_df.merge(item_features, on='item_id', how='left')
        data['price'] = data['sales_value']/data['quantity']
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)

        result = data.groupby(['user_id', 'sub_commodity_desc'])['price'].mean().reset_index()
        result.rename(columns={'price': 'mean_sum_purchase_sub_comm_desc'}, inplace=True)

        return result

    def _mean_sum_purchase_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Средняя сумма покупки 1 товара в каждой commodity_desc"""

        data = purchase_df.merge(item_features, on='item_id', how='left')
        data['price'] = data['sales_value']/data['quantity']
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)

        result = data.groupby(['user_id', 'commodity_desc'])['price'].mean().reset_index()
        result.rename(columns={'price': 'mean_sum_purchase_comm_desc'}, inplace=True)

        return result


    def _mean_sum_purchase_department(self, purchase_df, item_features):
        """DataFrame для фичи Средняя сумма покупки 1 товара в каждом department"""

        data = purchase_df.merge(item_features, on='item_id', how='left')
        data['price'] = data['sales_value']/data['quantity']
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)

        result = data.groupby(['user_id', 'department'])['price'].mean().reset_index()
        result.rename(columns={'price': 'mean_sum_purchase_department'}, inplace=True)

        return result


    def _purchases_in_sub_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Количество покупок в каждой sub_commodity_desc"""

        data = purchase_df.merge(item_features, on='item_id', how='left')

        result = data.groupby(['user_id', 'sub_commodity_desc'])['quantity'].sum().reset_index()
        result.rename(columns={'quantity': 'purchases_in_sub_commodity_desc'}, inplace=True)

        return result

    def _purchases_in_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Количество покупок в каждой commodity_desc"""

        data = purchase_df.merge(item_features, on='item_id', how='left')

        result = data.groupby(['user_id', 'commodity_desc'])['quantity'].sum().reset_index()
        result.rename(columns={'quantity': 'purchases_in_commodity_desc'}, inplace=True)

        return result


    def _purchases_in_department(self, purchase_df, item_features):
        """DataFrame для фичи Количество покупок в каждом department"""

        data = purchase_df.merge(item_features, on='item_id', how='left')

        result = data.groupby(['user_id', 'department'])['quantity'].sum().reset_index()
        result.rename(columns={'quantity': 'purchases_in_department'}, inplace=True)

        return result

    def _purchase_frequency(self, purchase_df):
        """DataFrame для фичи Частотность покупок раз/месяц"""

        data = purchase_df.copy()

        data['month'] = (data['day']-1)//30 + 1

        data_grouped = data.groupby(['user_id', 'month'])['basket_id'].unique().reset_index()
        data_grouped['frequency'] = data_grouped.apply(lambda row: len(row['basket_id']), axis=1)

        result = data_grouped.groupby('user_id')['frequency'].mean().reset_index()

        return result

    def _weekend_purchases_frac(self, purchase_df):
        """DataFrame для фичи Доля покупок в выходные (кол-во айтемов в выходные/общее кол-во приобретенных айтемов)"""

        result = purchase_df.groupby('user_id')['quantity'].sum().reset_index()
        weekend_purchases = pd.concat([purchase_df.loc[purchase_df['day']%6==0], purchase_df.loc[purchase_df['day']%7==0]])
        weekend_purchases = weekend_purchases.groupby('user_id')['quantity'].sum().reset_index()
        weekend_purchases.rename(columns={'quantity': 'weekend_quantity'}, inplace=True)
        result = result.merge(weekend_purchases, on='user_id', how='left')
        result.fillna(0, inplace=True)
        result['weekend_purchases_frac'] = result['weekend_quantity']/result['quantity']
        result.drop(['quantity', 'weekend_quantity'], axis=1, inplace=True)

        return result
                
    def _mean_sum_purchases_per_month(self, purchase_df):
        """"DataFrame для фичи Средняя сумма покупок/месяц"""

        data = purchase_df.copy()

        data['month'] = (data['day']-1)//30 + 1
        result = data.groupby(['user_id', 'month'])['sales_value'].sum().reset_index().groupby('user_id')['sales_value'].mean().reset_index()
        result.rename(columns={'sales_value': 'mean_sum_purchases_per_month'}, inplace=True)

        return result

    def _mean_quantity_per_basket(self, purchase_df):
        """DataFrame для фичи Среднее количество(quantity) покупаемых айтемов за одну покупку"""

        result = purchase_df.groupby(['user_id', 'basket_id'])['quantity'].sum().reset_index().groupby('user_id')['quantity'].mean().reset_index()
        result.rename(columns={'quantity': 'mean_quantity_per_basket'}, inplace=True)

        return result

    def _mean_purchases(self, purchase_df):
        """DataFrame для фичи Кол-во покупок в неделю"""

        result = purchase_df.groupby(['item_id', 'week_no'])['quantity'].sum().reset_index().groupby('item_id')['quantity'].mean().reset_index()
        result.rename(columns={'quantity': 'mean_purchases'}, inplace=True)

        return result

    def _mean_item_purchases_per_sub_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Среднее кол-во покупок 1 товара в sub_commodity_desc в неделю"""

        data = purchase_df.merge(item_features, on='item_id', how='left')

        result = data.groupby(['sub_commodity_desc', 'week_no'])['quantity'].sum().reset_index().groupby('sub_commodity_desc')['quantity'].mean().reset_index()
        result.rename(columns={'quantity': 'mean_item_purchases_per_sub_comm_desc'}, inplace=True)

        return result

    def _mean_item_purchases_per_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Среднее кол-во покупок 1 товара в commodity_desc в неделю"""

        data = purchase_df.merge(item_features, on='item_id', how='left')

        result = data.groupby(['commodity_desc', 'week_no'])['quantity'].sum().reset_index().groupby('commodity_desc')['quantity'].mean().reset_index()
        result.rename(columns={'quantity': 'mean_item_purchases_per_comm_desc'}, inplace=True)

        return result

    def _item_price(self, purchase_df):
        """DataFrame для фичи Цена товара"""

        data = purchase_df.copy()
        data['price'] = data['sales_value']/data['quantity']
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)
        result = data.groupby('item_id')['price'].mean().reset_index()

        return result

    def _mean_price_in_sub_comm_desc(self, purchase_df, item_features):
        """DataFrame для фичи Средняя цена товара в sub_commodity_desc"""

        data = purchase_df.merge(item_features, on='item_id', how='left')

        data['price'] = data['sales_value']/data['quantity']
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)

        result = data.groupby('sub_commodity_desc')['price'].mean().reset_index()
        result.rename(columns={'price': 'mean_price_in_sub_comm_desc'}, inplace=True)

        return result

    def _purchases_department_diff(self, purchase_df, item_features):
        """DataFrame для фичи Количество покупок в каждом department конкретного юзера в неделю - Среднее кол-во покупок всеми юзерами в department в неделю"""

        data = purchase_df.merge(item_features, on='item_id', how='left')
        all_users = data.groupby(['department', 'week_no'])['quantity'].sum().reset_index().groupby('department')['quantity'].mean().reset_index()
        all_users.rename(columns={'quantity': 'all_users'}, inplace=True)
        result = data.groupby(['user_id', 'department', 'week_no'])['quantity'].sum().reset_index().groupby(['user_id', 'department'])['quantity'].mean().reset_index()
        result = result.merge(all_users, on='department', how='left')
        result['purchases_department_diff'] = result['quantity'] - result['all_users']
        result.drop(['quantity', 'all_users'], axis=1, inplace=True)

        return result


    def fit_transform(self, recommend_df, purchase_df, item_features, user_features, with_targets=False):
        
        item_features.columns = [col.lower() for col in item_features.columns]
        user_features.columns = [col.lower() for col in user_features.columns]
        if 'product_id' in item_features.columns:
            item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
        if 'household_key' in user_features.columns:
            user_features.rename(columns={'household_key': 'user_id'}, inplace=True)

        item_features.replace(' ', np.nan, inplace=True)
        item_features['curr_size_of_product'].fillna('Unknown', inplace=True)
        item_features['department'].fillna('GROCERY', inplace=True)
        item_features['commodity_desc'].fillna('BEERS/ALES', inplace=True)
        item_features['sub_commodity_desc'].fillna('BEERALEMALT LIQUORS', inplace=True)

        result = recommend_df.apply(lambda x: pd.Series(x['recommendations']), axis=1).stack().reset_index(level=1, drop=True)
        result.name = 'item_id'
        result = recommend_df[['user_id']].join(result)
        if with_targets:
            result['drop'] = 1
            targets = purchase_df[['user_id', 'item_id']].copy()
            targets['target'] = 1
            result = result.merge(targets, on=['user_id', 'item_id'], how='left')
            result['target'].fillna(0, inplace=True)
            result.drop('drop', axis=1, inplace=True)

        result = result.merge(item_features, on='item_id', how='left')
        result = result.merge(user_features, on='user_id', how='left')
        result.fillna('Uknown', inplace=True)

        mean_check = self._mean_check(purchase_df)
        result = result.merge(mean_check, on='user_id', how='left')

        mean_sum_purchase_sub_comm_desc = self._mean_sum_purchase_sub_comm_desc(purchase_df, item_features)
        result = result.merge(mean_sum_purchase_sub_comm_desc, on=['user_id', 'sub_commodity_desc'], how='left')

        mean_sum_purchase_comm_desc = self._mean_sum_purchase_comm_desc(purchase_df, item_features)
        result = result.merge(mean_sum_purchase_comm_desc, on=['user_id', 'commodity_desc'], how='left')

        mean_sum_purchase_department = self._mean_sum_purchase_department(purchase_df, item_features)
        result = result.merge(mean_sum_purchase_department, on=['user_id', 'department'], how='left')

        purchases_in_sub_comm_desc = self._purchases_in_sub_comm_desc(purchase_df, item_features)
        result = result.merge(purchases_in_sub_comm_desc, on=['user_id', 'sub_commodity_desc'], how='left')

        purchases_in_comm_desc = self._purchases_in_comm_desc(purchase_df, item_features)
        result = result.merge(purchases_in_comm_desc, on=['user_id', 'commodity_desc'], how='left')

        purchases_in_department = self._purchases_in_department(purchase_df, item_features)
        result = result.merge(purchases_in_department, on=['user_id', 'department'], how='left')

        purchase_frequency = self._purchase_frequency(purchase_df)
        result = result.merge(purchase_frequency, on='user_id', how='left')

        weekend_purchases_frac = self._weekend_purchases_frac(purchase_df)
        result = result.merge(weekend_purchases_frac, on='user_id', how='left')

        mean_sum_purchases_per_month = self._mean_sum_purchases_per_month(purchase_df)
        result = result.merge(mean_sum_purchases_per_month, on='user_id', how='left')

        mean_quantity_per_basket = self._mean_quantity_per_basket(purchase_df)
        result = result.merge(mean_quantity_per_basket, on='user_id', how='left')

        mean_purchases = self._mean_purchases(purchase_df)
        result = result.merge(mean_purchases, on='item_id', how='left') #тут могут быть Nan
        result.fillna(0, inplace=True) 

        mean_item_purchases_per_sub_comm_desc = self._mean_item_purchases_per_sub_comm_desc(purchase_df, item_features)
        result = result.merge(mean_item_purchases_per_sub_comm_desc, on='sub_commodity_desc', how='left') #и тут могут быть nan
        result.fillna(0, inplace=True) 

        mean_item_purchases_per_comm_desc = self._mean_item_purchases_per_comm_desc(purchase_df, item_features)
        result = result.merge(mean_item_purchases_per_comm_desc, on='commodity_desc', how='left')
        result.fillna(0, inplace=True)

        item_price = self._item_price(purchase_df)
        result = result.merge(item_price, on='item_id', how='left')
        result.fillna(result['price'].mean(), inplace=True)

        mean_price_in_sub_comm_desc = self._mean_price_in_sub_comm_desc(purchase_df, item_features)
        result = result.merge(mean_price_in_sub_comm_desc, on='sub_commodity_desc', how='left')
        result.fillna(result['mean_price_in_sub_comm_desc'].mean(), inplace=True)

        purchases_department_diff = self._purchases_department_diff(purchase_df, item_features)
        result = result.merge(purchases_department_diff, on=['user_id', 'department'], how='left')
        result.fillna(min(result['purchases_department_diff']), inplace=True)

        result['price/mean_price_sub_comm_desc'] = result['price']/result['mean_price_in_sub_comm_desc']
        result['mean_sum_sub_comm_desc-price'] = result['mean_sum_purchase_sub_comm_desc'] - result['price']
        result['mean_sum_comm_desc-price'] = result['mean_sum_purchase_comm_desc'] - result['price']
        result['mean_sum_department-price'] = result['mean_sum_purchase_department'] - result['price']

        return result