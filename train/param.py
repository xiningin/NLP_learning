import argparse

parser = argparse.ArgumentParser()

#Bag-of-Word 和 N_Gram中对split方法的选择
parser.add_argument("--split_method" , type=str , default='random_some' , help='choose the method for split in feature extracting(eg:random_some、random_all、ordered_some、ordered_all)')
#选取的数据数量
parser.add_argument("--max_item" , type=int , default=30000 , help='input the max_item for train if choose not use all data')
#选Bag-of-Word还是N_Gram
parser.add_argument("--feature_method" , type=str , default='Bag-of-Word' , help='choose the method of feature extracting(eg:Bag-of-Word、N-Gram)')
parser.add_argument("--dimension" , type=int , default=2 , help='N of N-Gram')
#选择激活函数是softmax的还是sigmoid的
parser.add_argument("--activation_function" , type=str , default='softmax' , help='choose the activation function for classification(eg:softmax、sigmoid)')
#选择训练策略
parser.add_argument("--train_strategy" , type=str , default='mini-batch' , help='choose the strategy for classifier train(eg:shuffle、batch、mini-batch)')
parser.add_argument("--batch_size" , type=int , default=100 , help='input the batch_size for classifier train if the strategy is mini-batch')
#输入训练epoch
parser.add_argument("--epoch" , type=int , default=1 , help='input the epoch for train')
#拆分train.tsv的验证集占比
parser.add_argument("--dev_rate" , type=float , default=0.2 , help='When splitting the train.tsv file, the proportion of validation sets')
#平衡采样
parser.add_argument("--balance_sample" , type=bool , default=False , help='Whether to select in final_train, randomly select the 2 with large number of samples')

args = parser.parse_args()

