from N_Gram import N_Gram
import numpy as np
import csv
import random
from Linear_Regression import model
from param import args

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 150  --epoch 30 
# 0.5381904395745226

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 150  --epoch 30 
# 0.5381904395745226

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 250  --epoch 30 这个batch_size设置得不行
# 0.5329680891964629

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 150  --epoch 30 --dimension 3 增加了N看上去还可以
# 0.548154555940023

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 150  --epoch 30 --dimension 4 确实高了点
# 0.5509419454056133

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 120  --epoch 35 --dimension 4
# 0.569043957452262

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 120  --epoch 35 --dimension 4 在N-Gram里面改了一下calc_matrix(改成统计gram出现次数了)
# 0.5617711136742278

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 120  --epoch 70 --dimension 4 寻找模型极限epoch了
# 0.5779828271177753 在第66个epoch的时候取到最大，感觉还没到极限，还能卷hhhh

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 400  --epoch 70 --dimension 4 每个epoch都用完了所有数据(之前都是被max_item最大值限制为30000的)
# 0.5893886966551326 在第69个epoch取最大 还能卷？？？这都训练8个小时嘞

# 后面考虑试试减少一点类比比较多的2的数据量，尽可能平均
# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 400  --epoch 70 --dimension 4 --balance_sample True
# 0.5415355717304532 57的时候best epoch

# python final_main.py --split_method 'random_all' --feature_method 'N-Gram' --batch_size 400  --epoch 125 --dimension 4 每个epoch都用完了所有数据
# 0.5937780340894527 在第122个epoch取最大 还能卷？？？这都训练13个小时嘞



with open('../data/train.tsv') as train_file:
    reader = csv.reader(train_file , delimiter='\t')
    data = list(reader)
data = data[1:] # 有表头
random.seed(416)
np.random.seed(416)
learning_rate = 27


if args.balance_sample:
    nd_data = np.array(data)
    sentiment_2_data = nd_data[nd_data[:,3] == '2']
    sentiment_2_not_data = nd_data[nd_data[:,3] != '2']

    sentiment_2_data_idx = random.sample(range(len(sentiment_2_data)) , 30000)
    data = np.concatenate((sentiment_2_data[sentiment_2_data_idx] , sentiment_2_not_data)).tolist()

feature = N_Gram(data , dimension=args.dimension , max_item=args.max_item)
feature.create_dict()
feature.calc_matrix()

classifier = model(len(feature.train) , 5 , feature.dict_len)
# if args.balance_sample:
#     nd_data = np.array(feature.y_train)
#     sentiment_2_idx = np.where(nd_data == 2)
#     sentiment_not_2_idx = np.where(nd_data != 2)

#     sentiment_2_idx = random.sample(sentiment_2_idx , 30000)
#     nd_data_idx = np.concatenate(sentiment_2_idx , sentiment_not_2_idx)
#     if args.train_strategy == 'mini-batch':
#         classifier.final_train(
#             feature.train_matrix[sentiment_2_idx], 
#             feature.y_train[sentiment_2_idx], 
#             feature.test_matrix,
#             feature.y_test,
#             learning_rate, 
#             args.epoch,
#             int(len(sentiment_2_idx)/args.batch_size), 
#             'mini-batch', 
#             args.batch_size,
#         )
#     else:
#         raise Exception('wrong feature extracting method!!!!')
# else:
#     if args.train_strategy == 'mini-batch':
#         classifier.final_train(
#             feature.train_matrix, 
#             feature.y_train, 
#             feature.test_matrix,
#             feature.y_test,
#             learning_rate, 
#             args.epoch,
#             int(len(data)/args.batch_size), 
#             'mini-batch', 
#             args.batch_size,
#         )
#     else:
#         raise Exception('wrong feature extracting method!!!!')

if args.train_strategy == 'mini-batch':
    classifier.final_train(
        feature.train_matrix, 
        feature.y_train, 
        feature.test_matrix,
        feature.y_test,
        learning_rate, 
        args.epoch,
        int(len(feature.train)/args.batch_size), 
        'mini-batch', 
        args.batch_size,
    )
else:
    raise Exception('wrong feature extracting method!!!!')






