import numpy as np
from PIL import Image
import os
import pickle

root1 = 'C:/Users/hb/Desktop/data/CIFAR100'
root2 = 'C:/Users/hb/Desktop/data/from'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


with open(root2 + '/meta', 'rb') as infile:
    data = pickle.load(infile, encoding='latin1')
    classes = data['fine_label_names']

# 클래스 별 폴더 생성
os.mkdir(root1 + '/train_image')
os.mkdir(root1 + '/test_image')
for name in classes:
    os.mkdir(root1 + '/train_image/{}'.format(name))
    os.mkdir(root1 + '/test_image/{}'.format(name))

# Trainset Unpacking
# data_batch 파일들 순서대로 unpacking
print('Unpacking Train File')

train_file = unpickle(root2 + '/train')
train_data = train_file[b'data']

# 10000, 3072 -> 10000, 3, 32, 32 형태로 변환
train_data_reshape = np.vstack(train_data).reshape((-1, 3, 32, 32))
# 이미지 저장을 위해 10000, 32, 32, 3으로 변환
train_data_reshape = train_data_reshape.swapaxes(1, 3)
train_data_reshape = train_data_reshape.swapaxes(1, 2)
# 레이블 리스트 생성
train_labels = train_file[b'fine_labels']
# 파일 이름 리스트 생성
train_filename = train_file[b'filenames']

# 50000개의 파일을 순차적으로 저장
for idx in range(50000):
    train_label = train_labels[idx]
    train_image = Image.fromarray(train_data_reshape[idx])
    # 클래스 별 폴더에 파일 저장
    train_image.save(root1 + '/train_image/{}/{}'.format(classes[train_label], train_filename[idx].decode('utf8')))
# -----------------------------------------------------------------------------------------
# Testset Unpacking
print('Unpacking Test File')
test_file = unpickle(root2 + '/test')

test_data = test_file[b'data']

# 10000, 3072 -> 10000, 3, 32, 32 형태로 변환
test_data_reshape = np.vstack(test_data).reshape((-1, 3, 32, 32))
# 이미지 저장을 위해 10000, 32, 32, 3으로 변환
test_data_reshape = test_data_reshape.swapaxes(1, 3)
test_data_reshape = test_data_reshape.swapaxes(1, 2)
# 레이블 리스트 생성
test_labels = test_file[b'fine_labels']
# 파일 이름 리스트 생성
test_filename = test_file[b'filenames']

# 10000개의 파일을 순차적으로 저장
for idx in range(10000):
    test_label = test_labels[idx]
    test_image = Image.fromarray(test_data_reshape[idx])
    # 클래스 별 폴더에 파일 저장
    test_image.save(root1 + '/test_image/{}/{}'.format(classes[test_label], test_filename[idx].decode('utf8')))

print('Unpacking Finish')