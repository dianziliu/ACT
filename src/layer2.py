import torch as t
from torch.nn import Module
import numpy as np
import torch.nn.init as init 

class PreferenceTransformer(Module):
    def __init__(self,K,emb_size):
        super().__init__()
        self.K=K
        # self.aspect_embs=aspect_embs
        # emb_size=aspect_embs.shape[-1]
        self.w1=t.nn.Parameter(t.Tensor(emb_size,1))
        self.w2=t.nn.Parameter(t.Tensor(emb_size,1))
        # 初始化参数
        init.xavier_uniform_(self.w1.data,0.1)
        init.xavier_uniform_(self.w2.data,0.1)
        pass

    def get_preference_scores(self,user_embs,aspect_fre):
        # 还是没有引入位置的分数
        # 计算分数的核心步骤
        scores1=t.einsum('nd,bd->bn',self.aspect_embs,user_embs)
        # 引入用户的具体偏好信息，但是怎么计算
        # scores2=t.sin(t.exp(aspect_fre)/t.pi)
        scores2=t.sigmoid(t.exp(aspect_fre/100))
        # scores2=t.sin(t.exp(aspect_fre/100)/t.pi)
        scores=scores1+scores2
        # scores=pre_aspect_embs.sum(-1).sum(-1)
        preference_scores,idx=t.topk(scores,k=self.K)
        part_embs=self.aspect_embs[idx]
        return part_embs,preference_scores

    def attention(self,embs):
        # GATv2的方式
        # [b,K,d]
        h1=embs[:,0,:]@self.w1
        h2=embs@self.w2

        h1=t.repeat_interleave(h1,self.K+1,1)
        # h2=t.repeat_interleave(t.unsqueeze(t.squeeze(h2,-1),dim=1),self.K+1,1)
        h=h1+t.squeeze(h2,-1)
        h=t.nn.functional.leaky_relu(h)
        return h
    def forward(self,aspect_embs,user_embs,user_nodes,aspect_fre):
        self.aspect_embs=aspect_embs

        # 分别用于区分用户偏好和用户在图中的嵌入
        part_embs,preference_scores=self.get_preference_scores(user_embs,aspect_fre)
        # preference_scores=t.repeat_interleave(t.unsqueeze(preference_scores,1),self.K,1)
        
        # 将用户嵌入加入
        part_embs1=t.concat([t.unsqueeze(user_nodes,1),part_embs],1)      
        atten_scores=self.attention(part_embs1)
        # 取第0位的嵌入作为输出
        # atten_scores=atten_scores[:,:,0]
        final_scores=atten_scores[:,1:]+preference_scores
        aggre_res=t.einsum('bkd,bk->bd',part_embs,final_scores)
        # Add&Norm
        aggre_res=aggre_res+user_nodes
        aggre_res=t.nn.functional.normalize(aggre_res,dim=1)
        return aggre_res


class MultiHeadPreferenceTransformer(Module):
    def __init__(self,emb_size,K,num_heads):
        super().__init__()
        self.K         = K
        self.emb_size  = emb_size
        self.num_heads = num_heads
        self.ATLayers=[PreferenceTransformer(K,emb_size) for _ in range(num_heads)]
        # emb_size=aspect_embs.shape[-1]
        self.ffn1=t.nn.Parameter(t.Tensor(emb_size*num_heads,emb_size))
        init.xavier_uniform_(self.ffn1.data,0.1)
        pass

    def forward(self,aspect_embs,user_embs,user_nodes,aspect_fre):
        self.aspect_embs=aspect_embs
        # 多头计算
        res=[]
        for layer in self.ATLayers:
            layer.cuda()
            head_out=layer(aspect_embs,user_embs,user_nodes,aspect_fre)
            res.append(head_out)
        trans_embs=t.concat(res,-1)

        # 
        temp1=trans_embs@self.ffn1
        trans_emb_out=t.nn.functional.normalize(temp1+user_nodes)
        return trans_emb_out

if __name__=="__main__":
    a=t.tensor(np.random.random([100,10]),dtype=t.float32)
    b=t.tensor(np.random.random([6,10]),dtype=t.float32)
    # model=PreferenceTransformer(a,20)
    model=MultiHeadPreferenceTransformer(a,20,4)
    model(b,b)
    # model(a,b,10)

