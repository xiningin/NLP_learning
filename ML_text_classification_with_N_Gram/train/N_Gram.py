import data_split
import numpy as np
from param import args
from tqdm import tqdm

class N_Gram:
    def __init__(self , my_data , dimension = 2 , max_item = 1000):
        self.max_item = max_item
        self.word_dict = dict()
        self.dict_len = 0
        self.dimension = dimension
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
        
    def create_dict(self):
        for dime in range(1 , self.dimension+1): # 从1-gram提取到N-grame
            for item in tqdm(self.train + self.test , desc=f'create gram dict : {self.dimension}-gram'):
                words = item[2].lower().split()
                for i in range(len(words)-dime +1):
                    tokenized_word = ' '.join(words[i : i+dime])
                    if tokenized_word not in self.word_dict:
                        self.word_dict[tokenized_word] = len(self.word_dict)
        self.dict_len = len(self.word_dict)
        self.train_matrix = np.zeros((len(self.train) , self.dict_len))
        self.test_matrix = np.zeros((len(self.test) , self.dict_len))

    def calc_matrix(self):
        for dime in range(1 , self.dimension+1):
            for i in range(len(self.train)):
                words = self.train[i][2].lower().split()
                for j in range(len(words)-dime+1):
                    tokenized_word = ' '.join(words[j : j+dime])
                    self.train_matrix[i][self.word_dict[tokenized_word]] = 1
            for i in range(len(self.test)):
                words = self.test[i][2].lower().split()
                for j in range(len(words)-dime+1):
                    tokenized_word = ' '.join(words[j : j+dime])
                    self.test_matrix[i][self.word_dict[tokenized_word]] = 1
                    


