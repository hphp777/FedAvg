import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/hb/Desktop/data/archive/images_001/images/00000001_000.png')

np.set_printoptions(threshold=sys.maxsize)

plt.imshow(img[:,:,0] - img[:,:,2])
plt.show()
