import pandas as pd

test_lst = []

with open('train_val_list.txt') as f:
    test_lst = f.read().splitlines()

df = pd.read_csv('train_df_main.csv')
test_df = pd.DataFrame(columns =['Image Index','Patient ID','Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation','FilePath'])
df.drop(['No Finding'], axis = 1, inplace = True) 
idx = 0

for row in df.itertuples():
    item = []
    if row[2] in test_lst:
        item.append(row[2])
        item.append(idx)
        for i in range(4,19):
            item.append(row[i])
        test_df.loc[idx] = item
        idx += 1

test_df.to_csv('train_val_df.csv')