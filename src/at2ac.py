import numpy as np
import os

def merge_atac(at,ac,offset):
    # 获取矩阵的形状
    m, n = at.shape

    # 初始化一个空矩阵用于存储交错合并后的结果
    # merged = np.zeros((m, 2*n), dtype=at.dtype)

    ac = ac +offset

    at = np.expand_dims(at,axis=2)
    ac = np.expand_dims(ac,axis=2)
    merged=np.concatenate([at,ac],axis=2)
    return merged


def a2ac_sub(at,ac,offset):
    unique_values = np.unique(ac)
    m, n = at.shape
    all_subseqs= []
    # 按类别划分为多个子序列
    for i in range(m):
        sub_at= at[i]
        sub_ac =ac[i]
        subsequences =[]
        for c in unique_values:
            tmp =sub_at[sub_ac == c]
            tmp.append(c+offset)
            subsequences.append(tmp)
        all_subseqs.extend(subsequences)
    all_subseqs = np.array(all_subseqs)
    return all_subseqs

modes ={'merge':merge_atac,
            'sub':a2ac_sub}

def atseq2ac(data_loader, at_seqs,at2ac,mode='merge'):
    """ mode = {'merge','sub'}"""
    path = os.path.join(data_loader.path,'{}_at_ac.npy'.format(mode))

    if os.path.exists(path):
        at_ac=np.load(path)
    else:
        offset1=data_loader.n_users+ data_loader.n_items
        ac_seqs = at2ac[at_seqs-offset1] +1 # 0 保留，用于padding
        offset = data_loader.word_num+1 
        func=modes[mode]
        at_ac=func(at_seqs,ac_seqs,offset)
    return at_ac



