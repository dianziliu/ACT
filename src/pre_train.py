import os
import numpy as np

from data_loader import GraphData

from torch.nn import Module

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn.models import GAT,GCN,GraphSAGE, GIN
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges






# 定义 GCN 模型
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, 1)  # 用于边的打分

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # 拼接节点对嵌入并计算分数
        row, col = edge_index
        z_concat = torch.cat([z[row], z[col]], dim=1)
        return torch.sigmoid(self.lin(z_concat)).view(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


# 生成负样本（随机节点对作为负样本）
def get_negative_samples(edge_index, num_nodes, num_neg_samples,device):
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        i, j = torch.randint(0, num_nodes, (2,))
        if i != j and not ((edge_index[0] == i) & (edge_index[1] == j)).any():
            neg_edges.append([i, j])
    return torch.tensor(neg_edges).t().to(device)

# 训练循环
def train(data,model,optimizer,device):
    model.train()
    optimizer.zero_grad()

    # 正样本和负样本
    pos_edge_index = data.train_pos_edge_index
    neg_edge_index = get_negative_samples(
        data.train_pos_edge_index, 
        data.num_nodes, 
        pos_edge_index.size(1),
        device
    )

    # 构建训练边和标签
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([torch.ones(pos_edge_index.size(1)), 
                             torch.zeros(neg_edge_index.size(1))]).to(device)

    # 模型预测和损失
    out = model(data.x, data.train_pos_edge_index, edge_label_index)
    loss = F.binary_cross_entropy(out, edge_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试模型
@torch.no_grad()
def test(data,model,pos_edge_index, neg_edge_index,device):
    model.eval()
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([torch.ones(pos_edge_index.size(1)), 
                             torch.zeros(neg_edge_index.size(1))]).to(device)
    out = model(data.x, data.train_pos_edge_index, edge_label_index)
    pred = out > 0.5
    acc = (pred == edge_labels).sum().item() / edge_labels.size(0)
    return acc


def pretrain_model(args,data_loader:GraphData,path,model_name):

    # model=SUPPORT_PRETRAIN_MODEL[model]

    SUPPORT_PRETRAIN_MODEL={
            'gat':GAT,
            'gcn':GCNLinkPredictor,
            'SAGE':GraphSAGE,
            'gin':GIN
            }


    edg_index,edg_value,nodes, =data_loader.ui_graph

    num_nodes =max(nodes)+1
    node_embs = torch.nn.Parameter(
        torch.normal(0,0.1,
                     [num_nodes,args.word_dim]
                     )
        )
    edge_index = torch.tensor(edg_index.transpose(1,0),dtype=torch.long)
    data =Data(x=node_embs,edge_index=edge_index)
    data = train_test_split_edges(data)

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCNLinkPredictor(args.word_dim, 16).to(device)
    model =SUPPORT_PRETRAIN_MODEL[model_name](args.word_dim, 16).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 
    best_acc= 0 
    best_para= None

    # 训练模型
    for epoch in range(100):
        loss = train(data,model,optimizer,device)
        if epoch % 10 == 0:
            neg_edge_index = get_negative_samples(data.val_pos_edge_index, 
                                                            data.num_nodes, 
                                                            data.val_pos_edge_index.size(1),
                                                            device)
            val_acc = test(data,model,data.val_pos_edge_index, neg_edge_index,device)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc>best_acc:
                best_para = data.x
        # 存储训练结果
    
    # 测试模型性能
    data.x = best_para
    neg_edge_index = get_negative_samples(data.test_pos_edge_index, 
                                                data.num_nodes,
                                                data.test_pos_edge_index.size(1),
                                                device)
    test_acc = test(data,model,data.test_pos_edge_index, neg_edge_index,device)
    print(f"Test Accuracy: {test_acc:.4f}")

    res = best_para.cpu().detach().numpy()
    np.save(path,res)
    return res


def load_pretrain_embedding(args,data_loader:GraphData,model_name):

    path = os.path.join(data_loader.path,'{}_pretrain_emb.npy'.format(model_name))

    if os.path.exists(path):
        pretrai_emb=np.load(path)
    else:
        pretrai_emb=pretrain_model(args,data_loader,path,model_name)
    return pretrai_emb
    