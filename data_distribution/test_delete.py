import os
from glob import glob

d_list = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/test/Hernia_*.png')
print(len(d_list))

for i in range(len(d_list)):
    os.remove(d_list[i])