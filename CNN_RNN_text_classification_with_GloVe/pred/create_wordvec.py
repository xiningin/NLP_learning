import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset , DataLoader
import random

# 看用哪个glove模型
EMBEDDING_DIM = 300
TEST_DATASET_PATH = '../data/test.tsv'
DATASET_PATH_30000 = '../data/ave_sample_30000.tsv'
DATASET_PATH_7000 = '../data/ave_sample_7000.tsv'
TRAIN_DATASET_PATH = '../data/train.tsv'
SENT_COL_NAME = 'Phrase'
LABEL_COL_NAME = 'Sentiment'

def read_data_for_test(dataset_path , sent_col_name):
    data = pd.read_csv(dataset_path , sep='\t')
    X = data[sent_col_name].values
    return X

def read_data_for_val(dataset_path , sent_col_name , label_col_name):
    data = pd.read_csv(dataset_path , sep='\t')
    X = data[sent_col_name].values
    y = data[label_col_name].values
    return X , y
    

class myDataset(Dataset):
    def __init__(self , sents):
        self.sents = sents

    def __getitem__(self , idx):
        return self.sents[idx]

    def __len__(self):
        return len(self.sents)
    
class myValDataset(Dataset):
    def __init__(self , sents , labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self , idx):
        return self.sents[idx] , self.labels[idx]

    def __len__(self):
        return len(self.sents)
    
def collate_fn(batch_data):
    sents = batch_data
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents , batch_first=True , padding_value=0)

    return torch.LongTensor(padded_sents) , torch.FloatTensor(sents_len)

def val_collate_fn(batch_data):
    sents , labels = zip(*batch_data)
    sents_len = [len(sent) for sent in sents]
    sents = [torch.LongTensor(sent) for sent in sents]
    padded_sents = pad_sequence(sents , batch_first=True , padding_value=0)

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

def create_word_embedding(word2id , vec_dim):
    print("==========start create origin word embedding")
    word_embeddings = torch.nn.init.xavier_normal_(torch.empty(len(word2id) , vec_dim))
    
    word_embeddings[0 , :] = 0 # <pad>
    
    return word_embeddings.float()

def make_test_dataLodaer(train_dataset_kind=0,
                    test_dataset_path=TEST_DATASET_PATH,
                    sent_col_name=SENT_COL_NAME,
                    batch_size=32):
    print("==========Load data to batch")
    if train_dataset_kind == 0:
        train_dataset_path = TRAIN_DATASET_PATH
    elif train_dataset_kind == 1:
        train_dataset_path = DATASET_PATH_30000
    elif train_dataset_kind == 2:
        train_dataset_path = DATASET_PATH_7000
    train_X = read_data_for_test(dataset_path=train_dataset_path,
                      sent_col_name=sent_col_name)
    test_X = read_data_for_test(dataset_path=test_dataset_path,
                      sent_col_name=sent_col_name)
    
    X_word_and_id = word_and_id()
    X_word_and_id.fit(train_X)
    test_X = X_word_and_id.words_to_ids(test_X)
    word_embedding = create_word_embedding(X_word_and_id.word2id, vec_dim=EMBEDDING_DIM)
    
    testDataset = myDataset(test_X)
    testDataLoader = DataLoader(
        testDataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    print("==========Load test data finished")
    return testDataLoader , len(word_embedding) , word_embedding

def make_val_dataLodaer(train_dataset_kind=0,
                    sent_col_name=SENT_COL_NAME,
                    label_col_name=LABEL_COL_NAME,
                    batch_size=32):
    print("==========Load data to batch")
    if train_dataset_kind == 0:
        train_dataset_path = TRAIN_DATASET_PATH
    elif train_dataset_kind == 1:
        train_dataset_path = DATASET_PATH_30000
    elif train_dataset_kind == 2:
        train_dataset_path = DATASET_PATH_7000
    train_X , train_y = read_data_for_val(dataset_path=train_dataset_path,
                      sent_col_name=sent_col_name,
                      label_col_name=label_col_name)
    
    # 每个类别都取2000个
    sample_idx = random.sample(range(5000) , 2000)
    index_of_label0 = [i for i , item in enumerate(train_y) if item == 0]
    index_of_label1 = [i for i , item in enumerate(train_y) if item == 1]
    index_of_label2 = [i for i , item in enumerate(train_y) if item == 2]
    index_of_label3 = [i for i , item in enumerate(train_y) if item == 3]
    index_of_label4 = [i for i , item in enumerate(train_y) if item == 4]

    sample_label_1 = [index_of_label0[i] for i in sample_idx]
    sample_label_2 = [index_of_label1[i] for i in sample_idx]
    sample_label_3 = [index_of_label2[i] for i in sample_idx]
    sample_label_4 = [index_of_label3[i] for i in sample_idx]
    sample_label_5 = [index_of_label4[i] for i in sample_idx]

    final_sample = sample_label_1 + sample_label_2 + sample_label_3 + sample_label_4 + sample_label_5

    val_X = train_X[final_sample]
    val_y = train_y[final_sample]
    
    X_word_and_id = word_and_id()
    X_word_and_id.fit(train_X)
    val_X = X_word_and_id.words_to_ids(val_X)
    word_embedding = create_word_embedding(X_word_and_id.word2id, vec_dim=EMBEDDING_DIM)
    
    valDataset = myValDataset(val_X , val_y)
    valDataLoader = DataLoader(
        valDataset,
        batch_size=batch_size,
        collate_fn=val_collate_fn
    )

    print("==========Load test data finished")
    return valDataLoader , len(word_embedding) , word_embedding