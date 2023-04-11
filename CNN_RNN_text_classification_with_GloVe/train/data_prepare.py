import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset , DataLoader
import torchtext.vocab as vocab
from param import args

# 看用哪个glove模型
EMBEDDING_DIM = 300
GLOVE_PATH = '../GloVe/glove.42B.300d.txt'
DATASET_PATH = '../data/train.tsv'
DATASET_PATH_30000 = '../data/ave_sample_30000.tsv'
DATASET_PATH_7000 = '../data/ave_sample_7000.tsv'
SENT_COL_NAME = 'Phrase'
LABEL_COL_NAME = 'Sentiment'

def read_data(dataset_path , sent_col_name , label_col_name):
    data = pd.read_csv(dataset_path , sep='\t')
    X = data[sent_col_name].values
    y = data[label_col_name].values
    return X , y

"""
尽可能平均，只把数量最多的中性取了30000份，保持了1，2，3三类尽量平均，
如果对0-5都平均，取太少数据模型不易训练
"""
def create_ave_sample(dataset_path , label_col_name):
    data = pd.read_csv(dataset_path , sep='\t')
    label_2 = data[data[label_col_name] == 2]
    label_other = data[data[label_col_name] != 2]
    label_2 = label_2.sample(30000)
    ave_sample = pd.concat([label_2 , label_other] , axis=0)
    ave_sample.to_csv('../data/ave_sample_30000.tsv' , sep="\t" , index=None)

"""
完全平均数据，全部都取到7000条数据
"""
def create_full_ave_sample(dataset_path , label_col_name):
    data = pd.read_csv(dataset_path , sep='\t')
    label_0 = data[data[label_col_name] == 0]
    label_1 = data[data[label_col_name] == 1]
    label_2 = data[data[label_col_name] == 2]
    label_3 = data[data[label_col_name] == 3]
    label_4 = data[data[label_col_name] == 4]

    label_0 = label_0.sample(7000)
    label_1 = label_1.sample(7000)
    label_2 = label_2.sample(7000)
    label_3 = label_3.sample(7000)
    label_4 = label_4.sample(7000)
    
    ave_sample = pd.concat([label_0 , label_1 , label_2 , label_3 , label_4] , axis=0)
    ave_sample.to_csv('../data/ave_sample_7000.tsv' , sep="\t" , index=None)


class myDataset(Dataset):
    def __init__(self , sents , labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self , idx):
        return self.sents[idx] , self.labels[idx]

    def __len__(self):
        return len(self.sents)

def collate_fn(batch_data):
    batch_data.sort(key=lambda data_pair: len(data_pair[0]), reverse=True) # 不排序后面RNN里面使用pack_padded_sequence函数包装会出错
    sents , labels = zip(*batch_data)
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents , batch_first=True , padding_value=0) # 自动用最长的一个为其他的padding，节省计算资源

    return torch.LongTensor(padded_sents) , torch.LongTensor(labels) , torch.FloatTensor(sents_len)

class word_and_id:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

    def fit(self , sent_list):
        vocab = set()
        for sent in sent_list:
            vocab.update(sent.split())
        word_list = ["<pad>" , "unk"] + list(vocab) # collate_fn中padded填充用的是无含义的<pad>
        self.word2id = {word : i for i , word in enumerate(word_list)}
        self.id2word = {i : word for i , word in enumerate(word_list)}
    
    def words_to_ids(self , sent_list):
        sent_ids = []
        unk = self.word2id["unk"]
        for sent in sent_list:
            sent_id = list(map(lambda x : self.word2id.get(x , unk) , sent.split()))
            sent_ids.append(sent_id)
        return sent_ids
    
def create_word_embedding(word2id , embedding_file_path , vec_dim):
    print("==========start create word embedding")
    word_embeddings = torch.nn.init.xavier_normal_(torch.empty(len(word2id) , vec_dim))

    if args.random_embedding == 1:
        return word_embeddings.float()
    
    word_embeddings[0 , :] = 0 # <pad>
    with open(embedding_file_path , "r" , encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            splited = line.split()
            if splited[0] in word2id:
                word_embeddings[word2id[splited[0]]] = torch.tensor(list(map(lambda x : float(x) , splited[1:])))
    
    return word_embeddings.float()

def make_dataLodaer(dataset_path=DATASET_PATH,
                    sent_col_name=SENT_COL_NAME,
                    label_col_name=LABEL_COL_NAME,
                    batch_size=32,
                    glove_path=(GLOVE_PATH),
                    debug=False):
    print("==========Load data to batch")
    if args.do_ave_sample == 1:
        dataset_path = DATASET_PATH_30000
    elif args.do_ave_sample == 2:
        dataset_path = DATASET_PATH_7000

    X , y = read_data(
        dataset_path=dataset_path,
        sent_col_name=sent_col_name,
        label_col_name=label_col_name
    )
    if debug:
        X , y = X[:100] , y[:100]
    
    X_word_and_id = word_and_id()
    X_word_and_id.fit(X)
    X = X_word_and_id.words_to_ids(X)
    word_embedding = create_word_embedding(X_word_and_id.word2id, embedding_file_path=glove_path, vec_dim=EMBEDDING_DIM)
    
    X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=416)
    
    train_dataset , val_dataset = myDataset(X_train , y_train) , myDataset(X_val , y_val)
    train_dataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    print("==========Load data finished")
    return train_dataLoader , val_dataLoader , len(word_embedding) , word_embedding
    
if __name__ == "__main__":
    create_ave_sample(dataset_path=DATASET_PATH,
                      label_col_name=LABEL_COL_NAME
                      )
    create_full_ave_sample(dataset_path=DATASET_PATH,
                      label_col_name=LABEL_COL_NAME
                      )
    print("finished!!!")
    