import numpy as np

class model:
    def __init__(self, type_num , feature_num):
        self.type_num = type_num # 分类总数
        self.feature_num = feature_num # 提取的特征数量(维度)
        self.W_matrix = np.random.randn(feature_num , type_num) # 逻辑回归的权重矩阵，参数
    
    def active_matrix(self , matrix): # predict阶段用于计算多个vector
        matrix -= np.max(matrix , axis=1 , keepdims=True)
        matrix = np.exp(matrix)
        matrix /= np.sum(matrix , axis=1 , keepdims=True)
        return matrix
    
    def prediction(self , matrix):
        prob = self.active_matrix(matrix.dot(self.W_matrix))
        return prob.argmax(axis=1)
    