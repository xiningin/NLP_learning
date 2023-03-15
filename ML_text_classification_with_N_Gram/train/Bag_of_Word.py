import numpy as np
import data_split
from param import args
from tqdm import tqdm

class Bag:
    def __init__(self , my_data , max_item):
        self.max_item = max_item
        self.word_bag = dict()
        self.bag_len = 0
        if args.split_method == 'random_some':
            self.train , self.test = data_split.random_split(
                my_data,
                test_rate = args.dev_rate,
                max_item = max_item
            )
        elif args.split_method == 'random_all':
            self.train , self.test = data_split.all_data_random_split(
                my_data,
                test_rate = args.dev_rate,
            )
        elif args.split_method == 'ordered_some':
            self.train , self.test = data_split.ordered_split(
                my_data,
                test_rate = args.dev_rate,
                max_item = max_item
            )
        elif args.split_method == 'ordered_all':
            self.train , self.test = data_split.all_data_ordered_split(
                my_data,
                test_rate = args.dev_rate,
            )
        else:
            raise Exception("wrong split method!!!")
        
        self.y_train = [int(item[3]) for item in self.train]
        self.y_test = [int(item[3]) for item in self.test]
        self.train_matrix = None
        self.test_matrix = None
    
    def create_bag(self): #得全部转化为小写，其中标点符号不会受其影响
        for item in tqdm(self.train , desc='creating word bags'):
            phrase = item[2].lower()
            words = phrase.split()
            for word in words:
                if word not in self.word_bag:
                    self.word_bag[word] = len(self.word_bag) # 从0开始的
            self.bag_len = len(self.word_bag)
            self.train_matrix = np.zeros((len(self.train) , self.bag_len))
            self.test_matrix = np.zeros((len(self.test) , self.bag_len))
    
    def calc_matrix(self):      
        for i in range(len(self.train)):
            words = self.train[i][2].lower().split()
            for word in words:
                self.train_matrix[i][self.word_bag[word]] = 1
        for i in range(len(self.test)):
            words = self.test[i][2].lower().split()
            for word in words:
                self.test_matrix[i][self.word_bag[word]] = 1

