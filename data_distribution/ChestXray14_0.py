import torch
import pandas as pd
import os
from glob import glob

all_xrays_df = pd.read_csv('./train_df.csv')

all_image_paths = {os.path.basename(x): x for x in 
                   glob('C:/Users/hb/Desktop/Data/archive/images_*/images/*.png')}

my_file = open("test_list.txt", "r")
data = my_file.read()
testdata = data.replace('\n', ' ').split(" ")

aa = glob('C:/Users/hb/Desktop/Data/archive/images_*/images/*.png')
print(len(aa))
# If the finding label == disease_name, move to the designated folder
all_xrays_df['FilePath'] = all_xrays_df['Image Index'].map(all_image_paths.get)

all_xrays_df.to_csv('./train_df_main.csv')

# is_edema = all_xrays_df['Finding Labels'].str.contains('Edema')
# Edema = all_xrays_df[is_edema]
# indices = Edema.index
# all_xrays_df = all_xrays_df.drop(indices)
# Edema.to_csv('./data_list/Edema.csv')

# is_Emphysema = all_xrays_df['Finding Labels'].str.contains('Emphysema')
# Emphysema = all_xrays_df[is_Emphysema]
# indices = Emphysema.index
# all_xrays_df = all_xrays_df.drop(indices)
# Emphysema.to_csv('./data_list/Emphysema.csv')


# is_hernia = all_xrays_df['Finding Labels'].str.contains('Hernia')
# Hernia = all_xrays_df[is_hernia]
# indices = Hernia.index
# all_xrays_df = all_xrays_df.drop(indices)
# Hernia.to_csv('./data_list/Hernia.csv')

# is_Nofinding = all_xrays_df['Finding Labels'].str.contains('No Finding')
# Nofinding = all_xrays_df[is_Nofinding]
# indices = Nofinding.index
# all_xrays_df = all_xrays_df.drop(indices)
# Nofinding.to_csv('./data_list/NoFinding.csv')


# is_infiltration = all_xrays_df['Finding Labels'].str.contains('Infiltration')
# Infiltration = all_xrays_df[is_infiltration]
# indices = Infiltration.index
# all_xrays_df = all_xrays_df.drop(indices)
# Infiltration.to_csv('./data_list/Infiltration.csv')

# is_mass = all_xrays_df['Finding Labels'].str.contains('Mass')
# Mass = all_xrays_df[is_mass]
# indices = Mass.index
# all_xrays_df = all_xrays_df.drop(indices)
# Mass.to_csv('./data_list/Mass.csv')

# is_nodule = all_xrays_df['Finding Labels'].str.contains('Nodule')
# Nodule = all_xrays_df[is_nodule]
# indices = Nodule.index
# all_xrays_df = all_xrays_df.drop(indices)
# Nodule.to_csv('./data_list/Nodule.csv')

# is_Pleural_Thickening = all_xrays_df['Finding Labels'].str.contains('Pleural_Thickening')
# Pleural_Thickening = all_xrays_df[is_Pleural_Thickening]
# indices = Pleural_Thickening.index
# all_xrays_df = all_xrays_df.drop(indices)
# Pleural_Thickening.to_csv('./data_list/Pleural_Thickening.csv')

# is_Pneumothorax = all_xrays_df['Finding Labels'].str.contains('Pneumothorax')
# Pneumothorax = all_xrays_df[is_Pneumothorax]
# indices = Pneumothorax.index
# all_xrays_df = all_xrays_df.drop(indices)
# Pneumothorax.to_csv('./data_list/Pneumothorax.csv')



print(all_xrays_df.head())

# df = all_xrays_df['Finding Labels'].str.contains('Atelectasis')
# multiple = all_xrays_df[df]
# multiple.to_csv('./multiple_disease.csv')

# indices = multiple.index
# print(indices)
# all_xrays_df.drop(indices)

# is_atelectasis = all_xrays_df['Finding Labels'].str.contains('Atelectasis')
# Atelectasis = all_xrays_df[is_atelectasis]
# indices = Atelectasis.index
# all_xrays_df = all_xrays_df.drop(indices)
# Atelectasis.to_csv('./data_list/Atelectasis.csv')

# is_Pneumonia = all_xrays_df['Finding Labels'].str.contains('Pneumonia')
# Pneumonia = all_xrays_df[is_Pneumonia]
# indices = Pneumonia.index
# all_xrays_df = all_xrays_df.drop(indices)
# Pneumonia.to_csv('./data_list/Pneumonia.csv')

# is_cardiomegaly = all_xrays_df['Finding Labels'].str.contains('Cardiomegaly')
# Cardiomegaly = all_xrays_df[is_cardiomegaly]
# indices = Cardiomegaly.index
# all_xrays_df = all_xrays_df.drop(indices)
# Cardiomegaly.to_csv('./data_list/Cardiomegaly.csv')

# is_consolidation = all_xrays_df['Finding Labels'].str.contains('Consolidation')
# Consolidation = all_xrays_df[is_consolidation]
# indices = Consolidation.index
# all_xrays_df = all_xrays_df.drop(indices)
# Consolidation.to_csv('./data_list/Consolidation.csv')

# is_Effusion = all_xrays_df['Finding Labels'].str.contains('Effusion')
# Effusion = all_xrays_df[is_Effusion]
# indices = Effusion.index
# all_xrays_df = all_xrays_df.drop(indices)
# Effusion.to_csv('./data_list/Effusion.csv')

# is_fibrosis = all_xrays_df['Finding Labels'].str.contains('Fibrosis')
# Fibrosis = all_xrays_df[is_fibrosis]
# indices = Fibrosis.index
# all_xrays_df = all_xrays_df.drop(indices)
# Fibrosis.to_csv('./data_list/Fibrosis.csv')

