本次任务我使用PyTorch和GloVe词向量，实现了TextCNN和TextRNN两种模型，使用TextCNN论文中方法以及参数实现TextCNN，使用RNN、LSTM、Bi-LSTM实现模型TextRNN，并通过实验探究模型一些超参数。后续将训练得到的4个模型在Kaggle上进行预测评分，结果并不理想
### TextCNN
> 该部分使用平行的卷积size为3，4，5的二维卷积层(单词数量 * dim)
### TextRNN
> 该部分使用了RNN、LSTM、Bi-LSTM设计了三种模型，可以通过终端输入"rnn_type"选择，详细见train/param.py
### 需要注意的点
> 1.RNN变长pading冲淡了句子语义，不一定取最后一个隐藏变量hn,对此使用torch.nn.utils.rnn的pack_padded_sequence和pad_packed_sequence包装padding后的句子
> 2.torch.nn的交叉熵损失计算函数里面有一个softmax，所以传入loss之前不需要再softmax了，连续使用两次softmax会训练不稳定
> 3.collate_fn中可以sample上将长度较近的batch到一起，并且每一个batch填充到其中最大的长度，减少显存消耗


