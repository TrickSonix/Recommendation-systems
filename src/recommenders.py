import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, item_features, weighting=True):
                
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        item_features = item_features.loc[item_features['item_id'].isin(self.user_item_matrix.columns.values)]
        self.item_id_to_ctm = dict(zip(item_features['item_id'].tolist(), item_features['brand'].replace({'National': 0, 'Private': 1}).tolist()))
        self.item_id_to_ctm[999999] = 0
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, values='quantity', index='user_id', columns='item_id', aggfunc='count', fill_value=0)

        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        res = []
        top_n = self.top_purchases.loc[self.top_purchases['user_id'] == user]['item_id'].tolist()[:N]
        if filter_ctm:
            ctm_items = [item_id for item_id in self.item_id_to_ctm if self.item_id_to_ctm[item_id] == 1]
            minimum_items = len(self.userid_to_id)-len(ctm_items)+1
            for value in top_n:
                recs = self.model.similar_items(self.itemid_to_id[value], N=minimum_items)
                for rec in recs:
                    if self.id_to_itemid[rec[0]] in ctm_items:
                        res.append(self.id_to_itemid[rec[0]])
                        break
                    else:
                        continue
        else:
            for value in top_n:
                recs = self.model.similar_items(self.itemid_to_id[value], N=3)
                if self.id_to_itemid[recs[1][0]] != 999999:  
                    res.append(self.id_to_itemid[recs[1][0]])
                else:
                    res.append(self.id_to_itemid[recs[2][0]])

        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N)

        similar_users = [item[0] for item in similar_users]
        res = []
        for u in similar_users:
            user_item = [self.id_to_itemid[item[0]] for item in self.own_recommender.recommend(userid=u, user_items=csr_matrix(self.user_item_matrix).tocsr(), N=1, 
                                                                                        filter_already_liked_items=False, 
                                                                                        filter_items=[self.itemid_to_id[999999]],
                                                                                        recalculate_user=False )]
            res.extend(user_item)

        return res