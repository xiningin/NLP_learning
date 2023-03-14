import numpy as np
from tqdm import tqdm

class N_Gram:
    def __init__(self , my_data , dimension = 2):
        self.word_dict = dict()
        self.dict_len = 0
        self.dimension = dimension
        self.data = my_data
        if len(self.data[0]) > 3:
            self.y_data = [int(item[3]) for item in self.data]
        
    def create_dict(self):
        for dime in range(1 , self.dimension+1): # 从1-gram提取到N-grame
            for item in tqdm(self.data , desc=f'create gram dict : {self.dimension}-gram'):
                words = item[2].lower().split()
                for i in range(len(words)-dime +1):
                    tokenized_word = ' '.join(words[i : i+dime])
                    if tokenized_word not in self.word_dict:
                        self.word_dict[tokenized_word] = len(self.word_dict)
        self.dict_len = len(self.word_dict)
                    


