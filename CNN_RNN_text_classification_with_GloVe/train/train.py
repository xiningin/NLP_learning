from torch import optim
import torch
from text_model import TextCNN , TextRNN
from data_prepare import make_dataLodaer
from param import args
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

EMBEDDING_DIM = 300

if __name__ == "__main__":
    train_dataLodaer , val_dataLoader , vocab_size , vocab_vec = make_dataLodaer(batch_size=args.batch_size)
    model_name = args.model
    learning_rate = args.learning_rate
    epoch_num = args.epoch_num
    hidden_size4RNN = args.hidden_size4RNN
    dropout = args.dropout
    NUM_OF_CLASS = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "CNN":
        model = TextCNN(vocab_size=vocab_size , embedding_dim=EMBEDDING_DIM , num_of_class=NUM_OF_CLASS , embedding_vectors=vocab_vec , dropout=dropout , device=device)
    elif model_name in ["RNN" , "LSTM" , "Bi-LSTM"]:
        model = TextRNN(vocab_size=vocab_size , embedding_dim=EMBEDDING_DIM , hidden_size=hidden_size4RNN , num_of_class=NUM_OF_CLASS , embedding_vectors=vocab_vec , dropout=dropout , rnn_type=model_name , device=device)
    else:
        raise Exception('wrong model!!!!')

    optimizer = optim.Adam(model.parameters() , lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    train_accs = []
    val_accs = []
    train_losses = []
    best_model = model
    best_epoch = 0
    best_val_acc = 0
    best_loss = 1e6
    for epoch in range(epoch_num):
        train_acc = []
        val_acc = []

        print(f"=========={epoch+1} epoch train")
        model.train()
        for batch in tqdm(train_dataLodaer):
            X , y , sent_len = batch
            X = X.to(device)
            y = y.to(device)
            # sent_len = sent_len.to(device) sent_len需要时在cpu里面
            if model_name == "CNN":
                logits = model(X)
            else:
                logits = model(X , sent_len)
            optimizer.zero_grad()
            _, y_pre = torch.max(logits, -1)
            train_acc.append(torch.mean(y_pre == y , dtype=float).item())

            loss = loss_function(logits , y)
            if loss.item() < best_loss:
                best_loss = loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"=========={epoch+1} epoch eval in val_dataLoader")
        model.eval()
        for batch in tqdm(val_dataLoader):
            X , y , _ = batch
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            _ , y_pre = torch.max(logits , -1)
            val_acc.append(torch.mean(y_pre == y , dtype=float).tolist())
        
        train_accs.append(sum(train_acc)/len(train_acc))
        val_acc_ave = sum(val_acc)/len(val_acc)
        if val_acc_ave > best_val_acc:
            best_val_acc = val_acc_ave
            best_model = model
            best_epoch = epoch+1
        val_accs.append(val_acc_ave)

    
    plt.figure(figsize=(14,6))
    plt.plot(train_accs , color='aquamarine' , linestyle='dashdot', label='train_accs')
    plt.plot(val_accs , color='plum' , linestyle='solid',  label='val_accs')
    plt.title(f'{model_name}-train results')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    # now = datetime.now().strftime('%Y-%m-%d %H:%M')
    plt.savefig('../img/{}-{}-{}-{}-trainPlot.png'.format(args.do_ave_sample , model_name , round(best_val_acc,4) , hidden_size4RNN))
    plt.show()

    torch.save({'model':best_model.state_dict()} , '../model_param/{}-{}_42B_300d_glove_RELU.pth'.format(args.do_ave_sample , model_name))

    


