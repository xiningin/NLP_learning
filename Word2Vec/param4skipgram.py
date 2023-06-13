import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--loss_fun" , type=str , default='target' , help='Select loss function used in Skip_Gram(target , mine)')

args = parser.parse_args()

