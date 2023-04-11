import argparse

parser = argparse.ArgumentParser()

# 选择使用的模型
parser.add_argument("--model" , type=str , default="CNN" , help='choose the model to classification(CNN , RNN , LSTM , Bi-LSTM)')
# 输入学习率
parser.add_argument("--learning_rate" , type=float , default=0.001 , help='input the learning rate for train')
# 输入训练epoch数
parser.add_argument("--epoch_num" , type=int , default=30 , help='input the epoch num to train')
# 输入RNN或者LSTM网络隐藏层数
parser.add_argument("--hidden_size4RNN" , type=int , default=192 , help='input the hidden layer nums for RNN or LSTM')
# 输入训练batch_size
parser.add_argument("--batch_size" , type=int , default=64 , help='input batch_size for train')
# 选择是否随机初始化embedding
parser.add_argument("--random_embedding" , type=int , default=0 , help='choose whether use random embedding(0 , 1)')
# 输入dropout比率
parser.add_argument("--dropout" , type=float , default=0.5 , help='input the dropout rate for models')
# 设置平均采样
parser.add_argument("--do_ave_sample" , type=int , default=0 , help='choose whether use average sample(0 , 1 , 2)')

args = parser.parse_args()