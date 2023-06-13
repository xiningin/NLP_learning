import numpy as np
from data_utils import *

class InputData:
    def __init__(self):
        self.data_util = StanfordSentiment()
        self.tokens = self.data_util.tokens()
        self.token_freq = self.data_util._tokenfreq
        self.vocab_size = len(self.tokens)
        self.word2idx = {word : idx for idx , word in enumerate(self.tokens)}
        self.idx2word = {idx : word for idx , word in enumerate(self.tokens)}
        self.sample_table = self.data_util.sampleTable() # 负样本采样表
        self.batch_reader_ptr = 0
        self.all_sentence = self.data_util.allSentences()
        # self.all_sentence_length = self.data_util._cumsentlen[-1]
        self.sentence_num = len(self.all_sentence)

    def init_batch_reader_ptr(self):
        self.batch_reader_ptr = 0

    def get_posi_batch_pairs(self , batch_size , window_size):
        batch_pairs = []
        while len(batch_pairs) < batch_size:
            sentence = self.all_sentence[self.batch_reader_ptr]
            # print(sentence)
            word_ids = []
            for word in sentence:
                try:
                    word_ids.append(self.word2idx[word])
                except:
                    continue
            for i , focus in enumerate(word_ids):
                for j , context in enumerate(word_ids[max(0,i-window_size):i+window_size]):
                    assert focus < self.vocab_size
                    assert context < self.vocab_size
                    if i == j:
                        continue
                    batch_pairs.append((focus , context))

            self.batch_reader_ptr += 1
            if self.batch_reader_ptr == len(self.all_sentence):
                self.init_batch_reader_ptr()
        return batch_pairs

    def get_neg_batch_pairs(self , posi_word_pair , neg_num):
        neg_context = np.random.choice(
            self.sample_table , size=(len(posi_word_pair) , neg_num)
        ).tolist()
        return neg_context
    
    def calc_pair_num(self , window_size):
        all_num = 0
        for i in range(self.sentence_num):
            sentence = self.all_sentence[i]
            all_num += self.calc_one_sentence(sentence , window_size)
        return all_num
    
    def calc_one_sentence(self , sentence , window_size):
        sentence_len = len(sentence)
        pair_num = 0
        for i in range(sentence_len):
            pair_num += min(i , window_size) + min(sentence_len - i , window_size)
        return pair_num
    
if __name__ == '__main__':
    a = InputData()
    print(a.vocab_size)
    print(a.calc_pair_num(5))