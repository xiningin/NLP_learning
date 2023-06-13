import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from param4skipgram import args

class SkipGram(nn.Module):
    def __init__(self , vocab_size , embedding_dim):
        super(SkipGram , self).__init__()
        self.focus_embedding = nn.Embedding(vocab_size , embedding_dim , sparse=True)
        self.context_embedding = nn.Embedding(vocab_size , embedding_dim , sparse=True)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 初始化embedding参数，貌似是个trick，都这么写的
        init_range = 0.5 / self.embedding_dim
        self.focus_embedding.weight.data.uniform_(-init_range , init_range) # 生成的权重 -init_range <= 值 <= init_range
        self.context_embedding.weight.data.uniform_(-0 , 0)

    def forward(self , focus_idx , context_idx , neg_context_idx):
        focus_embedding = self.focus_embedding(focus_idx)
        context_embedding = self.context_embedding(context_idx)
        posi_score = torch.mul(focus_embedding , context_embedding) # 元素逐个相乘
        posi_score = torch.sum(posi_score , dim=1)
        posi_predi = F.logsigmoid(posi_score).squeeze() # 应该预测为1

        neg_context_embedding = self.context_embedding(neg_context_idx) # n个context_idx对应的embedding
        # focus_embedding.unsqueeze(2): n(广播)*embedding_dim*1 , neg_context_embedding: n*1*embedding_dim
        neg_score = torch.bmm(neg_context_embedding , focus_embedding.unsqueeze(2)).squeeze() # 压缩之前是n*1的二维
        neg_score = torch.sum(neg_score , dim=1)
        neg_predi = F.logsigmoid(neg_score).squeeze() # 应该预测为0

        # 不能loss向抵消，都得是正向的
        one = torch.LongTensor([1]).to(torch.device('cuda'));
        zero = torch.LongTensor([0]).to(torch.device('cuda'));
        loss = torch.square(one - posi_predi).sum() + torch.square(neg_predi - zero).sum()

        if args.loss_fun == 'mine':
            posi_score_2 = torch.mul(focus_embedding , context_embedding).squeeze()
            posi_score_2 = torch.sum(posi_score , dim=1)
            posi_score_2 = F.logsigmoid(posi_score_2)
            neg_score_2 = torch.bmm(neg_context_embedding , focus_embedding.unsqueeze(2)).squeeze()
            neg_score_2 = F.logsigmoid(-1 * neg_score_2)

            return -1 * (torch.sum(posi_score_2)+torch.sum(neg_score_2))
        return loss

    def save_embedding(self , file_name , use_cuda=True):
        if use_cuda:
            embedding = self.focus_embedding.weight.cpu().data.numpy()
        else:
            embedding = self.focus_embedding.weight.data.numpy()
        np.savetxt(file_name,embedding,delimiter='\t')
        print("csv data stored !!!")
    
    def load_focus_embedding(self , embedding):
        print((self.vocab_size , self.embedding_dim))
        assert embedding.shape == (self.vocab_size , self.embedding_dim)
        self.focus_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))

    def get_word_embed(self , word_idx):
        return self.focus_embedding.weight.data.numpy()[word_idx]


        
        
if __name__ == '__main__':
    # model = SkipGram(100 , 10)
    # focus = torch.LongTensor([1,2,3,4])
    # context = torch.LongTensor([10,12,13,14])
    # neg_context_idx = torch.LongTensor([[20,22],[22,31],[24,41],[26,33]])
    # print(model(focus , context , neg_context_idx))
    # a = torch.LongTensor([[1,2,1],[3,4,1]])
    # b = torch.LongTensor([[1,2,1],[3,4,1]])
    # print(torch.mul(a,b))
    # print(torch.LongTensor([1]) - torch.Tensor([0.1,0.2,0.3]))
    # model = SkipGram(100 , 10)
    # id2word = dict()
    # for i in range(100):
    #     id2word[i] = str(i)
    # model.save_embedding(id2word , 'test_demo.txt')

    model = nn.Embedding(100 , 10 , sparse=True)
    a = model.weight.data.numpy()
    np.savetxt('demo.csv',a,delimiter='\t') #frame: 文件 array:存入文件的数组

