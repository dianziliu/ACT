import argparse
import os
import random
import sys
from time import time
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
import numpy as np
import torch as t
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from data_loader import GraphData
# from model import ASG
# from model2 import ASG
# from src3.model import ASG  
from model import ACG
sys.path.append(".")
# os.chdir(os.path.dirname(__file__))

import numpy as np
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
# from torch_summary import model_summary 
import pandas as pd

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





def train_model(args,data_loader,model):
    print('################Training model###################')
    
    optimizer=optim.SGD(model.parameters(), lr=args.lr,
                        weight_decay=args.l2_re) 
    stopping_step = 0
    best_cu = None

    maes =[]
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        train_mse = train(args,data_loader,model,optimizer,epoch)
        if np.isnan(train_mse):
            exit(0)
        t2 = time()
        test_mse,test_mae = test(data_loader,model,dtype='eval')
        print('epoch{: d}: train_time:{: .2f}s, train_mse:{: .4f}; test_mse:{: .4f},test_rmse:{: .4f}.test_mae:{:.4f}'.format(
            epoch, t2 - t1, train_mse, test_mse,np.math.sqrt(test_mse),test_mae))
        best_cu, stopping_step, should_stop = early_stopping(
            test_mse, best_cu, test_mse, stopping_step, flag_step=5)
        maes.append(test_mae)
        # if should_stop:
        #     break
    print('best mse: {:.4f}\t best mae: {:.4f}'.format(test_value_r,min(maes)))

    # save 
    # scaffolding(args.dataset,Best_inputs,'best',data_generator.edge2idx)
    # scaffolding(args.dataset,Bad_inputs,'bad',data_generator.edge2idx)
    return test_value_r,min(maes)

def train(args,data_loader,model,optimizer,epoch):
    model.train()
    epoch_mse = 0.0
    inputs=[]

    for uid_batch, iid_batch, rate_batch in tqdm(data_loader.generate_batch(data_type='train'),ncols=100):
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
  
 
        
        preds= model(uid_batch, iid_batch,
                    False,' ')
        if epoch<0:
            
            loss = loss_function2(preds, rate_batch)
        else:
            loss = loss_function1(preds, rate_batch)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_mse += loss.item()
        # t2=time()
        # print(t2-t1)
    train_mse = epoch_mse / data_loader.train_length

    return train_mse


def test(data_loader,model,dtype):
    model.eval()
    epoch_loss = 0.0
    epoch_mae = 0.0
    if dtype == 'eval':
        data_length = data_loader.eval_length
    elif dtype == 'test':
        data_length = data_loader.test_length
    else:
        sys.exit()
    n_batch=0

    for (uid_batch, iid_batch, rate_batch) in data_loader.generate_batch(data_type=dtype):
        
        # 准备输入数据
        uid_batch = gpu(uid_batch)
        iid_batch = gpu(iid_batch)
        rate_batch = gpu(rate_batch, is_long=False)
        


        preds= model(uid_batch, iid_batch,
                     False,' ')
        # preds= model(uid_batch, iid_batch, 
        #             batch_nodes,batch_edges,batch_edge_type,batch_idx,
        #             user_nodeidx,item_nodeidx,
        #             False,' ')
        # 计算loss
        loss = loss_function2(preds, rate_batch)

        mae_loss=MAELoss(preds,rate_batch)
        epoch_loss += loss.item()
        epoch_mae += mae_loss.item()
        n_batch+=1
    final_mse = epoch_loss / data_length
    final_mae = epoch_mae/ n_batch
    model.train()
    return final_mse,final_mae



