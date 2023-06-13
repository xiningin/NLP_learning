import torch
import torch.nn as nn
from word2vec_input import InputData
from Skip_Gram import SkipGram
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import numpy as np

class word2vec:
    def __init__(self , embedding_dim=150 , batch_size=500 , window_size=5 , epoch=1 , init_lr=0.1):
        self.data = InputData()
        self.embedding_size = len(self.data.tokens)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch = epoch
        self.init_lr = init_lr
        self.skip_gram_model = SkipGram(self.embedding_size , embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.skip_gram_model.to(self.device)
        self.optimizer = optim.SGD(self.skip_gram_model.parameters() , lr=self.init_lr)
        self.loss_to_plot = []
        self.lr_to_plot = []
    
    def train(self):
        pair_num = self.data.calc_pair_num(self.window_size)
        batch_num = int(self.epoch * pair_num / self.batch_size)
        # print(f'pair_num: {pair_num}')
        # print(f'batch_num: {batch_num}')
        process_bar = tqdm(range(batch_num))
        for i in process_bar:
            posi_pairs = self.data.get_posi_batch_pairs(self.batch_size , self.window_size)

            neg_context = self.data.get_neg_batch_pairs(posi_pairs , 5) # 对于1个正的选取5个负样本
            posi_focus = [pair[0] for pair in posi_pairs]
            posi_context = [pair[1] for pair in posi_pairs]

            neg_context = Variable(torch.LongTensor(neg_context)).to(self.device)
            posi_focus = Variable(torch.LongTensor(posi_focus)).to(self.device)
            posi_context = Variable(torch.LongTensor(posi_context)).to(self.device)

            self.optimizer.zero_grad()
            loss = self.skip_gram_model(posi_focus , posi_context , neg_context)
            loss.backward()
            self.optimizer.step()
            process_bar.set_description('Loss: %.8f , lr: %.6f' %(loss.item() , self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.init_lr * (1.0 - 1.0 * i / batch_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            self.loss_to_plot.append(loss.item())
            self.lr_to_plot.append(self.optimizer.param_groups[0]['lr'])

        now = time.strftime('%m-%d %H-%M-%S', time.localtime())
        model_name = 'SkipGram' + now + '.csv'
        self.skip_gram_model.save_embedding(model_name , torch.cuda.is_available())

    def plot_train_process(self):
        x1 = range(len(self.loss_to_plot))
        x2 = range(len(self.lr_to_plot))
        # 创建并设置画布大小和标题
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle('Train Process')
        # 绘制第一个子图
        ax1 = fig.add_subplot(121)
        ax1.set_title('Loss')
        ax1.plot(x1, self.loss_to_plot, marker='o', color='#FF7F0E')
        # 绘制第二个子图
        ax2 = fig.add_subplot(122)
        ax2.set_title('Learning rate')
        ax2.plot(x2, self.lr_to_plot, marker='x', color='#1F77B4')

        now = time.strftime('%m-%d %H-%M-%S', time.localtime())
        pic_name = './img/SkipGram' + now + '.png'
        plt.savefig(pic_name)

        # 显示图形
        plt.show()

    def load_embedding(self , model_path):
        weight = np.loadtxt("embedding\SkipGram03-23 17-53-40.csv" , delimiter='\t' , dtype=float)
        self.skip_gram_model.load_focus_embedding(weight)

    def get_word_embed(self , word_idx):
        return self.skip_gram_model.get_word_embed(word_idx)

if __name__ == '__main__':
    # ==================模型训练===================
    print(time.strftime('%m-%d %H:%M:%S', time.localtime()))
    model = word2vec()
    model.train()
    model.plot_train_process()

    # ==================读文件验证===================
    # model = word2vec()
    # model.load_embedding("embedding\SkipGram03-23 17-53-40.csv")
    # print(model.get_word_embed(13))






