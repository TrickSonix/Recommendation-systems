import numpy as np

def hit_rate(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate

def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    top_recommend = np.array(recommended_list[:k])
    mask = np.isin(bought_list, top_recommend)
    hit_rate = (mask.sum() > 0)*1
    return hit_rate

def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    
    return precision

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    top_recommend = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    top_prices_recommended = np.array(prices_recommended[:k])
    
    mask = np.isin(top_recommend, bought_list)
    
    precision = mask.dot(top_prices_recommended)/top_prices_recommended.dot(np.ones(k))
    return precision

def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall

def recall_at_k(recommended_list, bought_list, k=5):
    
    top_recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    
    mask = np.isin(bought_list, top_recommended_list)
    recall = mask.sum()/len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    top_recommended_list = np.array(recommended_list[:k])
    bought_list = np.array(bought_list)
    top_prices_recommended = np.array(prices_recommended[:k])
    prices_bought = np.array(prices_bought)
    
    mask = np.isin(top_recommended_list, bought_list)
    
    recall = mask.dot(top_prices_recommended)/prices_bought.dot(np.ones(len(prices_bought)))
    return recall

def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result

def map_k(recommended_list, bought_list, k=5):
    
    #recommended_list и bought_list - списки списков рекомендаций и покупок пользователей
    assert len(recommended_list) == len(bought_list)
    result = 0
    for rec_list, b_list in zip(recommended_list, bought_list):
        result += ap_k(rec_list, b_list, k)
    
    return result