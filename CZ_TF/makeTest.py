import pandas as pd

test_lst = []

with open('test_list.txt') as f:
    test_lst = f.readlines()

df = pd.read_csv('train_df_main.csv')

