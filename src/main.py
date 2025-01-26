import os
import argparse

import numpy as np

from data_loader import GraphData
from pre_train import load_pretrain_embedding
from aspect_filter import aspect_filter
from train import train_model
from model import ACG
from at2ac import atseq2ac


def get_parase():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='yelp2', help='type of dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of batch of data')
    parser.add_argument('--num_layers', type=int,
                        default=2, help='number of GAT layer')
    parser.add_argument('--dim', type=int, default=32,
                        help='dim of user/item embedding')
    parser.add_argument('--word_dim', type=int, default=32,
                        help='dim of word embedding')
    parser.add_argument('--hidd_dim', type=int, default=32,
                        help='dim of graph node in TGAT')
    parser.add_argument('--factors', type=int, default=32,
                        help='dim of bilinear interaction')  # 这一项没有用到

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--l2_re', type=float,
                        default=0.2, help='weight of l2')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epoch')
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='dropout rate')
    parser.add_argument('--alpha', type=float,
                        default=0.9, help='alpha of aspect pooling')
    # 
    parser.add_argument('--K', type=int,
                        default=100, help='alpha of aspect pooling')
    parser.add_argument('--num_head', type=int,
                        default=8, help='alpha of aspect pooling')
    parser.add_argument('--bias', type=float,
                        default=3.0, help='bias of FM')
    parser.add_argument('--scale', type=float,
                            default=0.01, help='scale of FM')
    parser.add_argument('--res_path', type=str,
                        default='', help='bias of FM')
    args = parser.parse_known_args()[0]
    return args



def main():

    ## 0. 读取训练参数
    args =get_parase()
    data_loader = GraphData(args)
    ## 1. 加载预训练的结果
    
    pretrain_emb = load_pretrain_embedding(args,data_loader,'gcn')
    

    ## 2. 方面词序列结果
    user_fre,item_fre=data_loader.build_aspect_fre()
    user_topk_aspect,item_topk_aspect= aspect_filter(data_loader,
                                                     pretrain_emb,
                                                     user_fre,
                                                     item_fre,
                                                     args.K)

    ## 3. 加载全部数据
    user_topk_aspect_ac = atseq2ac(data_loader,user_topk_aspect,data_loader.at2ac)
    item_topk_aspect_ac = atseq2ac(data_loader,item_topk_aspect,data_loader.at2ac)
    ## 4. 训练模型
    model = ACG(data_loader,
                args,
                pretrain_emb,
                user_topk_aspect_ac,
                item_topk_aspect_ac,
                data_loader.at2ac)
    model.to('cuda')
    mse,mae = train_model(args,data_loader,model)

    if args.res_path!='':
        with open(args.res_path,'a') as f:
            f.write("{} scale:{} Bias:{} K:{} lr:{} rg:{}, MSE:{:.4f} MAE:{:.4f}\n".format(
                args.dataset,args.scale,args.bias,args.K,args.lr,args.l2_re,mse,mae
            ))
    pass

if __name__=='__main__':
    main()