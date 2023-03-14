import numpy as np
import random
from param import args
from tqdm import tqdm
import copy
from datetime import datetime

class model:
    def __init__(self , sample_num , type_num , feature_num):
        self.sample_num = sample_num # 样本数量
        self.type_num = type_num # 分类总数
        self.feature_num = feature_num # 提取的特征数量(维度)
        self.W_matrix = np.random.randn(feature_num , type_num) # 逻辑回归的权重矩阵，参数
        # final_train才使用下面几个best
        self.best_matrix = np.random.randn(feature_num , type_num) 
        self.best_rate = 0
        self.best_epoch = 0
    
    def active_vector(self , vector):
        if args.activation_function == 'softmax': ## 针对数值溢出减去最大值得softmax
            exp = np.exp(vector - np.max(vector)) # 减去最大值防止指数溢出
            return exp / exp.sum()
        elif args.activation_function == 'sigmoid': ## Logistic函数 / Sigmoid
            exp = np.exp(-vector)
            return 1 / (1 + exp)
        else:
            raise Exception('wrong activation function')
    
    def active_matrix(self , matrix): # predict阶段用于计算多个vector
        if args.activation_function == 'softmax': ## 针对数值溢出减去最大值得softmax
            matrix -= np.max(matrix , axis=1 , keepdims=True)
            matrix = np.exp(matrix)
            matrix /= np.sum(matrix , axis=1 , keepdims=True)
            return matrix
        elif args.activation_function == 'sigmoid': ## Logistic函数 / Sigmoid
            matrix = np.exp(-matrix)
            res = 1 / (1 + matrix)
            return res
        else:
            raise Exception('wrong activation function')
    
    def one_hot_y(self , y):
        res = np.array([0] * self.type_num)
        res[y] = 1
        return res.reshape(-1 , 1)
    
    def prediction(self , matrix):
        prob = self.active_matrix(matrix.dot(self.W_matrix))
        return prob.argmax(axis=1)
    
    def calc_correct_rate(self , train_matrix , y_train , test_matrix , y_test):
        num_train = len(train_matrix)
        pred_train = self.prediction(train_matrix)
        train_correct = sum(y_train == pred_train) / num_train

        num_test = len(test_matrix)
        pred_test = self.prediction(test_matrix)
        test_correct = sum(y_test == pred_test) / num_test

        return train_correct , test_correct
    
    def train(self , x , y , lr , train_times , strategy='mini-batch' , mini_size = 100):
        if self.sample_num != len(x) or self.sample_num != len(y):
            raise Exception("Sample size does not match!!!")
        if strategy == 'mini-batch':
            k = -1
            index_sample = list(range(self.sample_num))
            for i in tqdm(range(train_times) , desc="mini-batch training"):
                gradients = np.zeros((self.feature_num , self.type_num))
                for j in range(mini_size):
                    if not k == -1:
                        index_sample.remove(k)
                        if len(index_sample) == 0: 
                            index_sample = list(range(self.sample_num))
                    k = random.sample(index_sample , 1)[0] # 随机选取一个索引
                    y_pred = self.active_vector(self.W_matrix.T.dot(x[k].reshape(-1 , 1)))
                    gradients += x[k].reshape(-1 , 1).dot((self.one_hot_y(y[k]) - y_pred).T) # 不能加些什么绝对值之类的
                self.W_matrix += lr * gradients
        elif strategy == 'shuffle':
            k = -1
            index_sample = list(range(self.sample_num))
            for i in tqdm(range(train_times) , desc="shuffle training"):
                if k != -1:
                    index_sample.remove(k)
                    if len(index_sample) == 0: # 在多个epoch时容易remove没了，需要重新填充
                        index_sample = list(range(self.sample_num))
                k = random.sample(index_sample , 1)[0]
                y_pred = self.active_vector(self.W_matrix.T.dot(x[k].reshape(-1 , 1)))
                gradients = x[k].reshape(-1 , 1).dot((self.one_hot_y(y[k]) - y_pred).T) 
                self.W_matrix += lr * gradients
        elif strategy == 'batch':
            for i in tqdm(range(train_times) , desc="batch training"):
                gradients = np.zeros((self.feature_num , self.type_num))
                for j in range(self.sample_num):
                    y_pred = self.active_vector(self.W_matrix.T.dot(x[j].reshape(-1 , 1)))
                    gradients += x[j].reshape(-1 , 1).dot((self.one_hot_y(y[k]) - y_pred).T) 
                self.W_matrix += lr / self.sample_num * gradients
        else:
            raise Exception("undefined strategy")
        
    def final_train(self , x , y , x_test , y_test , lr , epoch , train_times , strategy='mini-batch' , mini_size = 100):
        if self.sample_num != len(x) or self.sample_num != len(y):
            raise Exception("Sample size does not match!!!")
        if strategy == 'mini-batch':
            for i in range(epoch):
                print(f'epoch: {i+1}')
                index_sample = list(range(self.sample_num))
                k = -1
                for j in tqdm(range(train_times) , desc="mini-batch training"):
                    gradients = np.zeros((self.feature_num , self.type_num))
                    for l in range(mini_size):
                        if k != -1:
                            index_sample.remove(k)
                        k = random.sample(index_sample , 1)[0] # 随机选取一个索引
                        y_pred = self.active_vector(self.W_matrix.T.dot(x[k].reshape(-1 , 1)))
                        gradients += x[k].reshape(-1 , 1).dot((self.one_hot_y(y[k]) - y_pred).T) # 不能加些什么绝对值之类的
                    self.W_matrix += lr / mini_size * gradients

                _ , test_rate = self.calc_correct_rate(
                    x , y,
                    x_test , y_test
                )
                if test_rate > self.best_rate:
                    self.best_epoch = epoch + 1
                    self.best_matrix = self.W_matrix
                    self.best_rate = test_rate
                    
            print(f"best_epoch: {self.best_epoch} \n best_rate: {self.best_rate}")
            now = datetime.now().strftime('%Y-%m-%d %H:%M')
            np.savetxt(f'tmp/{now}-model.tsv', self.best_matrix, delimiter='\t')
        else:
            raise Exception("undefined strategy")