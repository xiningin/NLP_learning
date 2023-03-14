from Bag_of_Word import Bag
from N_Gram import N_Gram
import numpy as np
import csv
import random
from Linear_Regression import model
import matplotlib.pyplot as plt
from param import args

with open('../data/train.tsv') as train_file:
    reader = csv.reader(train_file , delimiter='\t')
    data = list(reader)
data = data[1:] # 有表头
random.seed(416)
np.random.seed(416)

train_rates = []
test_rates = []
# learning_rates = [0.01,0.1,1,2,3]
learning_rates = [26,27,28,29,30,31,32]

# learning_rates = [1,1.1,1.2,1.3,1.4,1.5]
# learning_rates = [3 , 3.1 , 3.2 , 3.3 , 3.4 , 3.5 , 5 , 5.2 , 5.5]
# learning_rates = [6,7,8,9,10,11,12]
# learning_rates = [12,15,17,19,20]
# learning_rates = [20 , 21 , 22 , 23 , 24 , 25] #最好就直接用20几的了
# learning_rates = [26 , 28 ,30 , 32 , 34 , 35] 
# learning_rates = [20 , 23 , 26 , 30 , 32 , 35] 
# learning_rates = [32 , 35 , 38 , 40 , 42 , 45] 
# learning_rates = [10,12,15,20,25,30,50]

# python main-lr.py --feature_method 'Bag-of-Word'
if args.feature_method == 'Bag-of-Word':
    feature = Bag(data , max_item=args.max_item)
    feature.create_bag()
    feature.calc_matrix()
    fig_title = ''

    for lr in learning_rates:
        classifier = model(len(feature.train) , 5 , feature.bag_len)
        if args.train_strategy == 'mini-batch':
            classifier.train(
                feature.train_matrix, 
                feature.y_train, 
                lr, 
                int(args.max_item/args.batch_size) * args.epoch, 
                'mini-batch', 
                args.batch_size)
            fig_title = 'bag-of-word mini-batch'
        elif args.train_strategy == 'batch':
            classifier.train(
                feature.train_matrix, 
                feature.y_train, 
                lr, 
                args.epoch, 
                'batch',
                args.batch_size)
            fig_title = 'bag-of-word batch'
        elif args.train_strategy == 'shuffle':
            classifier.train(
                feature.train_matrix, 
                feature.y_train, 
                lr, 
                args.max_item * args.epoch, 
                'shuffle',
                args.batch_size)
            fig_title = 'bag-of-word shuffle'
        else:
            raise Exception('wrong train strategy!!!!')
        # 训练用的train数据，dev数据是全部没见过的，可以防止过拟合
        train_rate , test_rate = classifier.calc_correct_rate(
            feature.train_matrix,feature.y_train,
            feature.test_matrix,feature.y_test
        )
        train_rates.append(train_rate)
        test_rates.append(test_rate)
    plt.figure(figsize=(10,6))
    plt.title(fig_title)
    plt.plot(learning_rates , train_rates , label='train')
    plt.plot(learning_rates , test_rates , label='test')
    # Add the data value on head of the bar
    for i , j in zip(learning_rates , test_rates):
        plt.text(i , j , '%.2f'%j , ha='center' , va='bottom' , fontsize=10)
    plt.legend() 
    plt.show()
    
# python main-lr.py --feature_method 'N-Gram'
elif args.feature_method == 'N-Gram':
    feature = N_Gram(data , dimension=args.dimension , max_item=args.max_item)
    feature.create_dict()
    feature.calc_matrix()
    fig_title = ''

    for lr in learning_rates:
        classifier = model(len(feature.train) , 5 , feature.dict_len)
        if args.train_strategy == 'mini-batch':
            classifier.train(
                feature.train_matrix, 
                feature.y_train, 
                lr, 
                int(args.max_item/args.batch_size) * args.epoch,  
                'mini-batch', 
                args.batch_size)
            fig_title = 'N-Gram mini-batch'
        elif args.train_strategy == 'batch':
            classifier.train(
                feature.train_matrix, 
                feature.y_train, 
                lr, 
                args.epoch, 
                'batch',
                args.batch_size)
            fig_title = 'N-Gram batch'
        elif args.train_strategy == 'shuffle':
            classifier.train(
                feature.train_matrix, 
                feature.y_train, 
                lr, 
                args.max_item * args.epoch, 
                'shuffle',
                args.batch_size)
            fig_title = 'N-Gram shuffle'
        else:
            raise Exception('wrong train strategy!!!!')
        # 训练用的train数据，dev数据是全部没见过的，可以防止过拟合
        train_rate , test_rate = classifier.calc_correct_rate(
            feature.train_matrix,feature.y_train,
            feature.test_matrix,feature.y_test
        )
        train_rates.append(train_rate)
        test_rates.append(test_rate)

    plt.figure(figsize=(10,6))
    plt.title(fig_title)
    plt.plot(learning_rates , train_rates , label='train')
    plt.plot(learning_rates , test_rates , label='test')
    # Add the data value on head of the bar
    for i , j in zip(learning_rates , test_rates):
        plt.text(i , j , '%.2f'%j , ha='center' , va='bottom' , fontsize=10)
    plt.legend() 
    plt.show()


else:
    raise Exception('wrong feature extracting method!!!!')





