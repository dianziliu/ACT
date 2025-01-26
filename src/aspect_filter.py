
import torch
import numpy as np

def sigmoid(x):
    """
    使用 NumPy 实现 Sigmoid 函数。
    
    参数:
        x: 标量、NumPy 数组或其他可广播对象
    
    返回:
        Sigmoid 函数作用后的结果
    """
    return 1 / (1 + np.exp(-x))

def get_preference_score(ui_embs,aspect_embs):
    return np.einsum('id,jk->ij',ui_embs,aspect_embs)

def get_fre_score(aspect_fre):
    return sigmoid(np.exp(aspect_fre/100))
    


def get_aspect_score(ui_embs,aspect_fre,aspect_embs):

    pre_score=get_preference_score(ui_embs,aspect_embs)
    fre_score= get_fre_score(aspect_fre)
    return pre_score+fre_score


def get_topk_aspect(K, off1, user_aspect_score):
    user_aspect_score_srt=np.argsort(user_aspect_score,axis=1)
    topKuser_aspect=user_aspect_score_srt[:,:K]+off1
    return topKuser_aspect

def aspect_filter(data_loader, pre_train_emb,user_fre,item_fre,K):
    off1 =data_loader.n_users+data_loader.n_items
    user_embs= pre_train_emb[:data_loader.n_users]
    item_embs = pre_train_emb[data_loader.n_users:off1]
    aspect_embs = pre_train_emb[off1:]

    user_aspect_score =get_aspect_score(user_embs,user_fre,aspect_embs)
    item_aspect_score =get_aspect_score(item_embs,item_fre,aspect_embs)

    topKuser_aspect= get_topk_aspect(K, off1, user_aspect_score)
    topKitem_aspect =get_topk_aspect(K,off1,item_aspect_score)

    return topKuser_aspect,topKitem_aspect