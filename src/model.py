import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.data import Data
import torch_geometric.data as gda
from torch_geometric.nn import global_mean_pool as gmp,GCNConv,GATConv
from torch.nn import TransformerEncoder,TransformerEncoderLayer


from collections import defaultdict
import pandas as pd


def gpu(batch, is_long=True,cuda=True):
    if is_long:
        batch = t.LongTensor(batch)
    else:
        batch = t.FloatTensor(batch)
    if cuda:
        batch = batch.cuda()
    return batch

class ACG(nn.Module):
    def __init__(self, data_loader, args,pre_train_emb,user_aspect_seq,item_aspect_sqe,at2ac,to_gpu=True):
        super().__init__()
        self.args = args
        self.device = t.device('cuda' if to_gpu else 'cpu')
        self.data_loader = data_loader
        self.user_embedding = nn.Embedding(data_loader.n_users, args.dim)
        self.item_embedding = nn.Embedding(data_loader.n_items, args.dim)
        self.item_start = data_loader.n_users
        self.word_start=data_loader.n_users+data_loader.n_items
        # self.word_embedding = nn.Embedding(data_loader.n_words, args.word_dim)
    
        self.word_embedding = nn.Embedding(data_loader.word_num+100, args.word_dim)

        self.word_embedding.weight.data[:pre_train_emb.shape[0]] = t.tensor(pre_train_emb).to(self.device)

        self.user_aspect = gpu(user_aspect_seq,cuda=to_gpu)
        self.item_aspect = gpu(item_aspect_sqe,cuda=to_gpu)

        MAX_num_relations=8

        self.linear = nn.Linear(2*args.word_dim,args.hidd_dim)
        self.gt = TransformerEncoder(
                    TransformerEncoderLayer(args.hidd_dim,args.num_head,batch_first=True) ,3
                )
        
        

        self.trans_u = nn.ModuleList(
            [nn.Linear(args.dim, args.hidd_dim) for _ in range(args.num_layers)])
        self.trans_i = nn.ModuleList(
            [nn.Linear(args.dim, args.hidd_dim) for _ in range(args.num_layers)])
        self.trans_w = nn.ModuleList(
            [nn.Linear(args.hidd_dim, args.dim) for _ in range(args.num_layers)])
        
 
        
        self.interaction_u = nn.Linear(args.dim, args.dim)
        self.interaction_i = nn.Linear(args.dim, args.dim)
        self.fm2 = FM_Layer(args,data_loader)

        # 
        init.xavier_uniform_(self.user_embedding.weight,.1)
        init.xavier_uniform_(self.item_embedding.weight,.1)
        init.xavier_uniform_(self.linear.weight,.1)

        



    def forward(self, uid_batch, iid_batch,
                sig,store_file):
        
        # u_adj_ind -- index
        # u_adj_tp  -- type
        
        # 嵌入用户id和物品id
        self.u_e = self.user_embedding(uid_batch)
        self.i_e = self.item_embedding(iid_batch)


        user_nodes = self.word_embedding(uid_batch.unsqueeze(1))
        item_nodes = self.word_embedding((iid_batch+self.item_start).unsqueeze(1))


        user_at_ac_seq = self.word_embedding(self.user_aspect[uid_batch])
        item_at_ac_seq = self.word_embedding(self.item_aspect[iid_batch])

        user_at_ac_seq = self.linear(t.flatten(user_at_ac_seq,start_dim=-2))
        item_at_ac_seq = self.linear(t.flatten(item_at_ac_seq,start_dim=-2))


        user_aspects_embs = t.concat([user_nodes,user_at_ac_seq],dim=1)
        item_aspects_embs = t.concat([item_nodes,item_at_ac_seq],dim=1)


        user_aspects_embs = self.gt(user_aspects_embs)
        item_aspects_embs = self.gt(item_aspects_embs)


        u_pool_e = user_aspects_embs[:,0]
        i_pool_e = item_aspects_embs[:,0]
        # 对应公式（9）
        user_rep = self.interaction_u(self.u_e)
        item_rep = self.interaction_i(self.i_e)
        # 
        user_vc = t.cat((user_rep, u_pool_e), -1)
        item_vc = t.cat((item_rep, i_pool_e), -1)

        pre_rate = self.fm2(user_vc, item_vc, uid_batch, iid_batch)
        return pre_rate

    def get_alphas(self):
        return self.alphas

class FM_Layer(nn.Module):
    def __init__(self, args, data_loader) -> None:
        super(FM_Layer, self).__init__()
        # input_dim = (args.num_layers + 1) * args.dim * 2
        input_dim   =args.word_dim*4
        # input_dim = args.dim * 2
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.V = nn.Parameter(
            t.zeros(input_dim, input_dim), requires_grad=True)
        self.bias_u = nn.Parameter(
            t.zeros(data_loader.n_users, requires_grad=True))
        self.bias_i = nn.Parameter(
            t.zeros(data_loader.n_items, requires_grad=True))
        self.bias = nn.Parameter(t.ones(1)*args.bias)

        ### MI: 4.2 Train=False

        init.xavier_uniform_(self.V.data,0.1)

    def fm_layer(self, user_em, item_em, uid, iid):
        # linear_part: batch * 1 * input_dim
        x = 0.1 * t.cat((user_em, item_em), -1).unsqueeze(1)
        linear_part = self.linear(x).squeeze()
        batch_size = len(x)
        V = t.stack((self.V,) * batch_size)
        # batch * 1 * input_dim
        interaction_part_1 = t.bmm(x, V)  # (batch, 1, input_ dim)
        interaction_part_1 = t.pow(interaction_part_1, 2)
        interaction_part_2 = t.bmm(t.pow(x, 2), t.pow(V, 2))
        mlp_output = 0.5 * \
            t.sum((interaction_part_1 - interaction_part_2).squeeze(1), -1)
        rate = linear_part + mlp_output + \
            self.bias_u[uid] + self.bias_i[iid] +  self.bias
        # rate = linear_part + mlp_output + self.bias
        
        return rate

    def forward(self, user_em, item_em, uid, iid):
        return self.fm_layer(user_em, item_em, uid, iid).view(-1)
        # return self.ncf_layer(user_em,item_em,uid,iid).view(-1)



class NCF_Layer(nn.Module):
    def __init__(self, args, config):
        super(NCF_Layer, self).__init__()
        self.user_embedding =nn.Embedding(config['n_users'],args.word_dim)
        self.item_embedding =nn.Embedding(config['n_items'],args.word_dim)
        self.linear= nn.Linear(3*args.word_dim,1)
        self.linear2= nn.Linear(4*args.word_dim,args.word_dim,bias=False)
        ### MI: 4.2 Train=False

        init.normal_(self.user_embedding.weight,.0,.1)
        init.normal_(self.item_embedding.weight,.0,.1)
        init.normal_(self.linear.weight,.0,.1)


    def ncf_layer(self,user_em,item_em,uid,iid):
        x =  t.cat((user_em, item_em), -1)
        x = self.linear2(x) 
        x = t.cat([self.user_embedding(uid),
                   self.item_embedding(iid),
                   x],-1)
        rate = self.linear(x)
        return rate


    def forward(self, user_em, item_em, uid, iid):
        return self.ncf_layer(user_em,item_em,uid,iid).view(-1)


def store_res(file, uid_batch, iid_batch, 
              u_nodes, u_adj_ind, u_adj_tp, 
              i_nodes, i_adj_ind, i_adj_tp,
              u_alpha_res,i_alpha_res,edge2idx):
    # 1. 将adj_ind 转化回去，从local变为global


    user_df = train2df(uid_batch, u_nodes, u_adj_ind, u_alpha_res,'user',edge2idx,u_adj_tp)
    item_df = train2df(iid_batch, i_nodes, i_adj_ind, i_alpha_res,'item',edge2idx,i_adj_tp)
    
    return user_df,item_df

def train2df(id_batch, nodes, adj_ind, aplha_res,side,edge2idx,adj_tp):

    #
    
    #
    adj_ind_ori=[]
    adj_ty_ori=[]
    for i in range(len(nodes)):
        this_node=nodes[i].cpu().detach().numpy()
        this_adj=adj_ind[i].cpu().detach().numpy()
        this_adj_tp=adj_tp[i].cpu().detach().numpy()
        xs=list(this_adj[0])
        ys=list(this_adj[1])
        adj_ind_ori.extend(
            [[this_node[x],this_node[y]] for x,y in  zip(xs,ys)]
        )
        adj_ty_ori.extend([list(
            edge2idx[x].cpu().numpy()) for x in this_adj_tp])
    # 2. 
    # for uid,edges, in zip(list(uid_batch.cpu().numpy()),):
    ids=[]
    for uid,u_edges, in zip(list(id_batch.cpu().numpy()),adj_ind):
        ids.extend([uid]*len(u_edges[0]))
    d=defaultdict(list)


    # 记录alpha
    for i in range(len(aplha_res)): # num_layes
            # alpha_res,out=aplha_res[i][j]
        res=aplha_res[i][0]
        # aplha_res[i][1] 是out
        for k,alpha in enumerate(res):
            d["{}_alpha_{}_{}".format(side,i,k)].extend(alpha)
    
    d['{}_id'.format(side)]=ids
    d['{}_edge_h'.format(side)]=[x[0] for x in  adj_ind_ori]
    d['{}_edge_t'.format(side)]=[x[1] for x in  adj_ind_ori]
    d['{}_edge_r'.format(side)]=adj_ty_ori
    # d['edge_r']=
    df=pd.DataFrame(d)
    return df