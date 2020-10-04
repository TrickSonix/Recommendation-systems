import numpy as np
import pandas as pd
import itertools

from src.utils import prefilter_items, postfilter_items
from src.recommenders import MainRecommender

class GridSearch():

    def __init__(self, recommender, top_n_list, weighting_list, model_params, scoring_func):
        self.recommender = recommender
        self.top_n_list = top_n_list
        self.weighting_list = weighting_list
        self.model_params = model_params
        self.scoring_func = scoring_func

    def _get_param_grid(self, param_grid):
        param_lists = []
        param_names = []
        for param in param_grid:
            param_lists.append(param_grid[param])
            param_names.append(param)

        param_grid_list = []
        for prod in itertools.product(*param_lists):
            param_dict = {}
            for name, value in zip(param_names, prod):
                param_dict[name] = value
            param_grid_list.append(param_dict)

        return param_grid_list

    def fit(self, data_train, data_val, item_sub_comm, prices, N=5):
        param_grid_list = self._get_param_grid(self.model_params)
        result_lvl_1 = data_val.groupby('user_id')['item_id'].unique().reset_index()
        result_lvl_1.columns = ['user_id', 'actual']
        best_params = []
        best_score = 0
        if self.recommender == 'MainRecommender':
            for top_n in self.top_n_list:
                X = prefilter_items(data_train, take_n_popular=top_n)
                for weighting in self.weighting_list:
                    model = MainRecommender(X, weighting=weighting, **param_grid_list[0])
                    result_lvl_1['recs'] = result_lvl_1['user_id'].apply(lambda x: model.get_main_model_recommendations(x, N=top_n))
                    result_lvl_1 = postfilter_items(result_lvl_1, 'recs', item_sub_comm, prices, N=N)
                    score = result_lvl_1.apply(lambda row: self.scoring_func(row['postfilter_recs'], row['actual'], k=200), axis=1).mean()
                    if score > best_score:
                        best_score = score
                        best_params = [top_n, weighting, param_grid_list[0]]
                    for param in param_grid_list[1:]:
                        model.model = model.fit(model.user_item_matrix, **param)
                        result_lvl_1['recs'] = result_lvl_1['user_id'].apply(lambda x: model.get_main_model_recommendations(x, N=top_n))
                        result_lvl_1 = postfilter_items(result_lvl_1, 'recs', item_sub_comm, prices, N=N)
                        score = result_lvl_1.apply(lambda row: self.scoring_func(row['postfilter_recs'], row['actual'], k=N), axis=1).mean()
                        if score > best_score:
                            best_score = score
                            best_params = [top_n, weighting, param]
            
            return best_score, best_params

