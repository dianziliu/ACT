from ast import arg
import numpy as np
import pickle
import sys
import itertools as itl
from random import shuffle
# itl.chain 相当于一个迭代器组合，
from time import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch as t

FLAGE_GPU=1

def gpu(batch, is_long=True):
    if is_long:
        batch = t.LongTensor(batch)
        # batch=t.IntTensor(batch)
    else:
        batch = t.FloatTensor(batch)
    if FLAGE_GPU:
        batch = batch.cuda()
    return batch


class GraphData():
    def __init__(self, args):
        # self.path = args.path
        # self.path = '../data/' + args.dataset + '/'
        self.path="data4/"+args.dataset + '/'
        self.args = args
        #wsss
        train_file       = self.path + 'data.train'
        eval_file        = self.path + 'data.eval'
        test_file        = self.path + 'data.test'
        para_file        = self.path + 'data.para'
        user_graphs_file = self.path + 'data.user_graphs'
        item_graphs_file = self.path + 'data.item_graphs'
        edge2idx_file    = self.path+"data.edge2idx"
        ui_graph_file    = self.path+'data.ui_graph'
        #
        self.train_data = self.load_data(train_file)
        self.eval_data  = self.load_data(eval_file)
        self.test_data  = self.load_data(test_file)
        #
        # self.test_data  = self.test_data[:int(len(self.test_data)/2)] 
        #
        self.para_data  = self.load_data(para_file)
        self.u_graphs   = self.load_data(user_graphs_file)
        self.i_graphs   = self.load_data(item_graphs_file)
        self.edge2idx   = self.load_data(edge2idx_file)
        self.ui_graph   = self.load_data(ui_graph_file)
        self.num_edge   = len(self.edge2idx)
        self.statistic_ratings()
        print('number of users:', self.n_users)
        print('number of items:', self.n_items)
        print('number of words:', self.word_num)

        self.input_datas={}
    def load_data(self, file):
        file_path = open(file, 'rb')
        data = pickle.load(file_path)
        file_path.close()
        return data

    def statistic_ratings(self):
        self.n_users      = self.para_data['user_num']
        self.n_items      = self.para_data['item_num']
        self.word2id      = self.para_data['vocab']
        self.word_num     = max(self.word2id.values())+1
        self.train_length = self.para_data['train_length']
        self.eval_length  = self.para_data['eval_length']
        self.test_length  = self.para_data['test_length']


    def generate_batch(self, data_type='train'):
        
        if data_type == 'train':
            np.random.shuffle(self.train_data)
            data = self.train_data
        elif data_type == 'eval':
            data = self.eval_data
        elif data_type == 'test':
            data = self.test_data
        else:
            sys.exit()
        

        iter_num = int(np.ceil(len(data) / self.args.batch_size))
        self.iter_num=iter_num
        # shuffle(data)
        for batch_num in range(iter_num):
            # 1. 数据分批
            start_index = batch_num * self.args.batch_size
            if start_index >= len(data):
                continue
            end_index = min((batch_num + 1) * self.args.batch_size, len(data))     
            batch_data = data[start_index:end_index]
            uid_batch, iid_batch, rate_batch = zip(*batch_data)

            # 2. 备份数据       
            if data_type not in self.input_datas:
                self.input_datas[data_type]={}
            if batch_num not in self.input_datas[data_type]:
                # uid_batch, iid_batch, rate_batch=batch[:,0],batch[:,1],batch[:,2]
                # 整理nodes，edges,edge_type
                u_nodes, i_nodes = [], []
                u_adj_ind, i_adj_ind = [], []
                u_edge_type, i_edge_type = [], []
                last_u=1
                batch_nodes=[]
                batch_edges=[]
                batch_edge_type=[]
                batch=[]
                batch_idx=0
                for i, user in enumerate(uid_batch):
                    if user not in self.u_graphs:
                        user=last_u
                    adj_ind, adj_val, node_index = self.u_graphs[user]
                    u_adj_ind.append(adj_ind)
                    u_nodes.append(node_index)
                    u_edge_type.append(adj_val)
                    batch_nodes.extend(node_index)
                    batch.extend([batch_idx]*len(adj_val))
                    batch_edges.extend(adj_ind)
                    batch_edge_type.extend(adj_val)
                    batch_idx+=1
                last_i=1
                for i, item in enumerate(iid_batch):
                    if item not in self.i_graphs:
                        item=last_i
                    adj_ind, adj_val, node_index = self.i_graphs[item]
                    i_adj_ind.append(adj_ind)
                    i_nodes.append(node_index)
                    i_edge_type.append(adj_val)
                    batch_nodes.extend(node_index)
                    batch.extend([batch_idx]*len(adj_val))
                    batch_edges.extend(adj_ind)
                    batch_edge_type.extend(adj_val)

                    batch_idx+=1
                # 去重
                batch_nodes=list(set(batch_nodes))
                batch_edges=np.array(batch_edges)
                # if check_mem():
                #     uid_batch       = gpu(uid_batch)
                #     iid_batch       = gpu(iid_batch)
                #     rate_batch      = gpu(rate_batch, is_long=False)
                #     batch_nodes     = gpu(batch_nodes)
                #     batch_edges     = gpu(batch_edges)
                #     batch_edge_type = gpu(batch_edge_type)
                #     batch           = gpu(batch)
                 # 这一步会消耗大量的内存
                self.input_datas[data_type][batch_num]=(batch_nodes, batch_edges,batch_edge_type,batch)           
            # batch_edges=np.array(batch_edges)
            batch_nodes, batch_edges,batch_edge_type,batch=self.input_datas[data_type][batch_num]


            yield uid_batch, iid_batch, rate_batch, batch_nodes, batch_edges,batch_edge_type,batch


    def build_aspect_fre(self):
        user_fre=np.zeros([self.n_users,len(self.word2id)])
        item_fre=np.zeros([self.n_users,len(self.word2id)])
        offset=self.n_users+self.n_items
        # 
        for user, (adj_ind, adj_val, node_index) in self.u_graphs.items():
            for i,j in adj_ind:
                if user!=i:
                    assert 'error'
                user_fre[user][j-offset]+=1
        #
        for item, (adj_ind, adj_val, node_index) in self.i_graphs.items():
            for i,j in adj_ind:
                if item-self.n_users!=i:
                    assert 'error'
                item_fre[item][j-offset]+=1

        self.user_fre=user_fre
        self.item_fre=item_fre
        return user_fre,item_fre
def check_mem(N=20):
    # 获取当前使用的GPU索引
    current_gpu_index = t.cuda.current_device()
    # 查看房前使用的现存
    used_memory = t.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
    if used_memory>=N:
        return False
    else:
        return True

