import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class TextCNN(nn.Module):
    def __init__(self , vocab_size , embedding_dim , num_of_class , embedding_vectors=None , out_channel=64 , kernel_size=[3,4,5] , dropout=0.5 , device="cpu"):
        super(TextCNN , self).__init__()
        if embedding_vectors is None:
            self.embed = nn.Embedding(vocab_size , embedding_dim).to(device)
        else:
            self.embed = nn.Embedding(vocab_size , embedding_dim , _weight=embedding_vectors).to(device)
        self.convs = nn.ModuleList([nn.Conv2d(1 , out_channel , (K , embedding_dim)).to(device) for K in kernel_size])
        self.dropout = nn.Dropout(dropout).to(device)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(kernel_size.__len__() * out_channel , num_of_class).to(device)
    
    def forward(self , x):
        # x:(batch_size , seq_len)
        embed_out = self.embed(x).unsqueeze(1) # (batch_size , seq_len , dim)
        conv_out = [F.relu(conv(embed_out)).squeeze(3) for conv in self.convs] # 3个(batch_szie , out_channel , 100-K+1)
        pool_out = [F.max_pool1d(block , block.size(2)).squeeze(2) for block in conv_out] # 3个(batch_size , out_channel)
        pool_out = torch.cat(pool_out , 1) # (batch_size , 3*out_channel)
        pool_out = self.relu(pool_out) # 加入一下激活
        pool_out = self.dropout(pool_out)
        logits = self.classifier(pool_out) # (batch_size , num_of_class)

        return logits

class TextRNN(nn.Module):
    def __init__(self , vocab_size , embedding_dim , hidden_size , num_of_class , embedding_vectors = None , dropout=0.2 , rnn_type="RNN" , device="cpu"):
        super(TextRNN , self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_of_class = num_of_class
        self.embedding_size = embedding_dim
        self.relu = nn.ReLU(inplace=False)
        self.rnn_type = rnn_type
        self.device = device
        if embedding_vectors is None:
            self.embed = nn.Embedding(vocab_size , embedding_dim).to(device)
        else:
            self.embed = nn.Embedding(vocab_size , embedding_dim , _weight=embedding_vectors).to(device)

        self.dropout = nn.Dropout(dropout).to(device)
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim , hidden_size , batch_first=True).to(device)
            self.classifier = nn.Linear(hidden_size , num_of_class).to(device)
        elif rnn_type == "LSTM":
            self.lstm = nn.LSTM(embedding_dim , hidden_size , batch_first=True , bidirectional=False).to(device)
            self.classifier = nn.Linear(hidden_size , num_of_class).to(device)
        elif rnn_type == "Bi-LSTM":
            self.lstm = nn.LSTM(embedding_dim , hidden_size , batch_first=True , bidirectional=True).to(device)
            self.classifier = nn.Linear(hidden_size*2 , num_of_class).to(device)
    
    def forward(self , input_sents , sents_len):
        batch_size , _ = input_sents.shape # (batch_size,seq_len)
        embed_out = self.embed(input_sents) # (batch_size,seq_len,dim)
        packed_inputs = rnn_utils.pack_padded_sequence(embed_out , sents_len , batch_first=True , enforce_sorted=False)

        if self.rnn_type == "RNN":
            h0 = torch.randn(1 , batch_size , self.hidden_size).to(self.device)
            packed_outputs , hn = self.rnn(packed_inputs , h0)
            outputs , _ = rnn_utils.pad_packed_sequence(packed_outputs , batch_first=True)
            # outputs = outputs[:,-1,:]
        elif self.rnn_type == "LSTM":
            h0 = torch.randn(1 , batch_size , self.hidden_size).to(self.device)
            c0 = torch.randn(1 , batch_size , self.hidden_size).to(self.device)
            packed_outputs,(hn,cn) = self.lstm(packed_inputs , (h0 , c0))
            outputs , _ = rnn_utils.pad_packed_sequence(packed_outputs , batch_first=True)
            # outputs = outputs.reshape(batch_size , -1)
        elif self.rnn_type == "Bi-LSTM":
            h0 = torch.randn(2 , batch_size , self.hidden_size).to(self.device)
            c0 = torch.randn(2 , batch_size , self.hidden_size).to(self.device)
            packed_outputs,(hn,cn) = self.lstm(packed_inputs , (h0 , c0)) # 用包含两个方向最后的结果来分类
            outputs , _ = rnn_utils.pad_packed_sequence(packed_outputs , batch_first=True)
            # outputs = outputs[:,-1,:]
        outputs = outputs[:,-1,:]
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs).squeeze(0)

        return logits


if __name__ == '__main__':
    # ===============CNN=================================
    # model = TextCNN(300 , 20 , 2)
    # print(model)
    # n_t = torch.randint(low=0, high=10, size=(3, 100))
    # print(model(n_t))
    # ===============RNN=================================
    model = TextRNN(300 , 20 , 50 , 3)
    print(model)
    seqs = torch.tensor([
        [1,2,5,6,2,0,0],
        [1,2,5,6,0,0,0],
        [1,0,0,0,0,0,0]
    ])
    lengths = torch.tensor([5,4,1])
    print(model(seqs , lengths))
    