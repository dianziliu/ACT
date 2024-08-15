import argparse
import os
import random
import sys
sys.path.append('.')
from time import time

import numpy as np
import torch as t
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from data_loader import GraphData

from model import AGT
sys.path.append(".")


import numpy as np
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
# from torch_summary import model_summary 
import pandas as pd
from sklearn.preprocessing import StandardScaler


from queue import Queue
from multiprocessing import Process
import multiprocessing
FLAGE_GPU=1


Best_loss=10000
# Best_inputs=None
Bad_loss=0
# Bad_inputs=None



if FLAGE_GPU:
    MAELoss=MeanAbsoluteError().cuda()
    loss_function1=t.nn.MSELoss(reduction='sum').cuda()
    loss_function2=t.nn.MSELoss(reduction='sum').cuda()
else:
    MAELoss=MeanAbsoluteError()
    loss_function1=t.nn.MSELoss(reduction='sum')
    loss_function2=t.nn.MSELoss(reduction='sum')

def model_summary(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def gpu(batch, is_long=True):
    if is_long:
        batch = t.LongTensor(batch)
    else:
        batch = t.FloatTensor(batch)
    if FLAGE_GPU:
        batch = batch.cuda()
    return batch

def gpu_int(batch,use_code=True):
    return t.IntTensor(batch)


def form_normlize(data):
    mx=data.max()
    mi=data.min()
    data=(data-mi)/(mx-mi)
    return data

def trans2gpu(q,data_generator,daty_type):
    for uid_batch, iid_batch, rate_batch, batch_nodes, batch_edges,\
        batch_edge_type,batch_idx,user_nodeidx,item_nodeidx in data_generator.generate_batch(data_type='train'):
        
        uid_batch = gpu(uid_batch)
        iid_batch = gpu(iid_batch)
        rate_batch = gpu(rate_batch, is_long=False)
        
        this_nodes=gpu(batch_nodes)
        this_edges=gpu(batch_edges).T
        this_edge_types=gpu(batch_edge_type)
        this_batch=gpu(batch_idx)

        this_user_nodeidx=gpu(user_nodeidx)
        this_item_nodedix=gpu(item_nodeidx)    
        q.pull([uid_batch, iid_batch, 
                    this_nodes,this_edges,this_edge_types,this_batch,
                    this_user_nodeidx,this_item_nodedix])

def early_stopping(log_value, best_value, test_mse, stopping_step, flag_step=3):
    # early stopping strategy:
    global test_value_r
    global test_value_m
    if best_value is None or log_value <= best_value:
        stopping_step = 0
        best_value = log_value
        test_value_r = test_mse
    else:
        stopping_step += 1
    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop





def train_model(args):
    print('################Training model###################')
    stopping_step = 0
    best_cu = None
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        train_mse = train(args,epoch)
        t2 = time()
        test_mse,test_mae = test(dtype='eval')
        print('epoch{: d}: train_time:{: .2f}s, train_mse:{: .4f}; test_mse:{: .4f},test_rmse:{: .4f}.test_mae:{:.4f}'.format(
            epoch, t2 - t1, train_mse, test_mse,np.math.sqrt(test_mse),test_mae))
        best_cu, stopping_step, should_stop = early_stopping(
            test_mse, best_cu, test_mse, stopping_step, flag_step=5)
        # if should_stop:
        #     break
    print('best mse: {:.4f}'.format(test_value_r))

    # save 
    # scaffolding(args.dataset,Best_inputs,'best',data_generator.edge2idx)
    # scaffolding(args.dataset,Bad_inputs,'bad',data_generator.edge2idx)

def train(args,epoch):
    model.train()
    epoch_mse = 0.0
    inputs=[]

    for uid_batch, iid_batch, rate_batch, batch_nodes, batch_edges,\
        batch_edge_type,batch_idx in tqdm(data_generator.generate_batch(data_type='train'),ncols=100):
    # for uid_batch, iid_batch, rate_batch, batch_nodes, batch_edges,\
    #     batch_edge_type,batch_idx in data_generator.generate_batch(data_type='train'):  
        optimizer.zero_grad()
        # preds= model(uid_batch, iid_batch, 
        #             batch_nodes,batch_edges,batch_edge_type,batch_idx,
        #             user_nodeidx,item_nodeidx,
        #             False,' ')
        uid_batch = gpu(uid_batch)
        iid_batch = gpu(iid_batch)
        rate_batch = gpu(rate_batch, is_long=False)
        
        this_nodes=gpu(batch_nodes)
        this_edges=gpu(batch_edges).T
        this_edge_types=gpu(batch_edge_type)
        this_batch=gpu(batch_idx)

 
        
        preds= model(uid_batch, iid_batch, 
                    this_nodes,this_edges,this_edge_types,this_batch,
                    False,' ')
        if epoch<3:
            
            loss = loss_function2(preds, rate_batch)
        else:
            loss = loss_function1(preds, rate_batch)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_mse += loss.item()
        # t2=time()
        # print(t2-t1)
    train_mse = epoch_mse / data_generator.train_length

    return train_mse


def test(dtype):
    model.eval()
    epoch_loss = 0.0
    epoch_mae = 0.0
    if dtype == 'eval':
        data_length = data_generator.eval_length
    elif dtype == 'test':
        data_length = data_generator.test_length
    else:
        sys.exit()
    n_batch=0

    for (uid_batch, iid_batch, rate_batch,batch_nodes, batch_edges,
         batch_edge_type,batch_idx) in data_generator.generate_batch(data_type=dtype):
        
        # 准备输入数据
        uid_batch = gpu(uid_batch)
        iid_batch = gpu(iid_batch)
        rate_batch = gpu(rate_batch, is_long=False)
        
        #
        this_nodes=gpu(batch_nodes)
        this_edges=gpu(batch_edges).T
        this_edge_types=gpu(batch_edge_type)
        this_batch=gpu(batch_idx)
  

        preds= model(uid_batch, iid_batch, this_nodes,
                     this_edges,this_edge_types,this_batch,
                     False,' ')
        # preds= model(uid_batch, iid_batch, 
        #             batch_nodes,batch_edges,batch_edge_type,batch_idx,
        #             user_nodeidx,item_nodeidx,
        #             False,' ')
        # 计算loss
        loss = loss_function2(preds, rate_batch)

        #
        global Best_inputs, Best_loss,Bad_loss,Bad_inputs
        if loss<Best_loss:  
            Best_loss=loss
            Best_inputs=[uid_batch, iid_batch, rate_batch,batch_nodes, batch_edges,
                         batch_edge_type,batch_idx]
            
        if loss>Bad_loss:
            Bad_loss=loss
            Bad_inputs=[uid_batch, iid_batch, rate_batch,batch_nodes, batch_edges,
                         batch_edge_type,batch_idx]



        mae_loss=MAELoss(preds,rate_batch)
        epoch_loss += loss.item()
        epoch_mae += mae_loss.item()
        n_batch+=1
    final_mse = epoch_loss / data_length
    final_mae = epoch_mae/ n_batch
    model.train()
    return final_mse,final_mae




def scaffolding(dataset,model_input,name,edge2idx):
    model.eval()
    (uid_batch, iid_batch, rate_batch,batch_nodes, batch_edges,
         batch_edge_type,batch_idx,user_nodeidx,item_nodeidx )=model_input
    
    # 准备输入数据
    # uid_batch = gpu(uid_batch)
    # iid_batch = gpu(iid_batch)
    # rate_batch = gpu(rate_batch, is_long=False)
    
    #
    this_nodes=gpu(batch_nodes)
    this_edges=gpu(batch_edges).T
    this_edge_types=gpu(batch_edge_type)
    this_batch=gpu(batch_idx)
    
    this_user_nodeidx=gpu(user_nodeidx)
    this_item_nodedix=gpu(item_nodeidx)

    # 
    preds= model(uid_batch, iid_batch, this_nodes,
                     this_edges,this_edge_types,this_batch,
                     this_user_nodeidx,this_item_nodedix,
                     True,' ')
        # 计算loss
    alphas=model.get_alphas()
    # loss = loss_function2(preds, rate_batch)
    df1=pd.DataFrame({
        'uid':uid_batch.cpu().tolist(),
        'iid':iid_batch.cpu().tolist(),
        'rate':rate_batch.cpu().tolist(),
        'pred':preds.cpu().tolist()
    })

    d2={}
    d2['h']=[x[0] for x in batch_edges]
    d2['t']=[x[1] for x in batch_edges]
    d2['r']=[edge2idx[x] for x in batch_edge_type]
    for i in range(args.num_layers):
        for idx,alpha in enumerate(alphas[i]):
            d2['alpha_{}_{}'.format(i,idx)]=list(alpha)
    
    df2=pd.DataFrame(d2)
    dir='src_ACG/v1_full/res/{}'.format(dataset)
    if not os.path.exists(dir):
        os.mkdir(dir)
    df1.to_csv(os.path.join(dir,'{}_data.csv'.format(name)),index=False)
    df2.to_csv(os.path.join(dir,'{}_kg.csv'.format(name)),index=False)


paths = [
    "Musical_Instruments_5",
    "Office_Products_5",
    "Toys_and_Games_5",
    "Video_Games_5",
    "Beauty_5",
    "yelp2",
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='Office_Products_5', help='type of dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of batch of data')
    parser.add_argument('--num_layers', type=int,
                        default=2, help='number of GAT layer')
    parser.add_argument('--dim', type=int, default=16,
                        help='dim of user/item embedding')
    parser.add_argument('--word_dim', type=int, default=16,
                        help='dim of word embedding')
    parser.add_argument('--hidd_dim', type=int, default=16,
                        help='dim of graph node in TGAT')
    parser.add_argument('--factors', type=int, default=16,
                        help='dim of bilinear interaction')  # 这一项没有用到

    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--l2_re', type=float,
                        default=1e-4, help='weight of l2')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epoch')
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='dropout rate')
    parser.add_argument('--alpha', type=float,
                        default=0.9, help='alpha of aspect pooling')
    # 
    parser.add_argument('--K', type=int,
                        default=300, help='alpha of aspect pooling')
    parser.add_argument('--num_head', type=int,
                        default=8, help='alpha of aspect pooling')
    args = parser.parse_known_args()[0]
    print(args.dataset)
    seed_number=2024

    np.random.seed(seed_number)
    random.seed(seed_number)
    t.manual_seed(seed_number)
    t.cuda.manual_seed_all(seed_number)
    # 
    data_generator = GraphData(args)
    test_value_r = 0
    loss_function = t.nn.MSELoss(reduction='sum')
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_words'] = data_generator.word_num

    user_fre,item_fre=data_generator.build_aspect_fre()
    # ss=StandardScaler()
    # user_fre_std=form_normlize(user_fre)
    # item_fre_std=form_normlize(item_fre)
    #
    model = AGT(config, args,gpu(data_generator.edge2idx),gpu(user_fre),gpu(item_fre))
    print( model_summary(model))
    if FLAGE_GPU:
        model = model.cuda()
   

    # optimizer = optim.Adam(model.parameters(), lr=args.lr,
    #                        weight_decay=args.l2_re)
    optimizer=optim.SGD(model.parameters(), lr=args.lr,
                        weight_decay=args.l2_re)                # 2023.5.8 新增rg的影响，之前的结果都是没有rg的        
    train_model(args)
