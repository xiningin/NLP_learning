import random
import copy

def random_split(data , test_rate = 0.3 , max_item = 1000): # max_item一般是接收main-lr中传的值(终端输入)
    train = []
    test = []

    test_num = int(max_item * test_rate)
    train_num = max_item - test_num

    copy_data = copy.deepcopy(data)

    train = random.sample(copy_data , train_num)
    for item in train:
        copy_data.remove(item)
    test = random.sample(copy_data , test_num)

    return train , test

def ordered_split(data , test_rate = 0.3 , max_item = 1000):
    train = []
    test = []

    if len(data) > max_item:
        tmp = round(max_item * (1 - test_rate))
    else:
        tmp = round(len(data) * (1 - test_rate))
    
    train = data[:tmp]
    test = data[tmp:max_item]

    return train , test

def all_data_random_split(data , test_rate = 0.2):
    train = []
    test = []

    test_num = int(len(data) * test_rate)
    train_num = len(data) - test_num

    copy_data = copy.deepcopy(data)

    train = random.sample(copy_data , train_num)
    for item in train:
        copy_data.remove(item)
    test = random.sample(copy_data , test_num)

    return train , test

def all_data_ordered_split(data , test_rate = 0.2):
    train = []
    test = []

    tmp = round(len(data) * (1 - test_rate))
    
    train = data[:tmp]
    test = data[tmp:]

    return train , test
