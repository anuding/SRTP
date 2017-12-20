#coding=utf-8
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.image as mpimg # mpimg
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

jinface = mpimg.imread('jin.jpg')
jinface = np.array(jinface)
jinface = jinface[:, :, 0]
jinface = jinface / 255.
plt.imshow(jinface)
plt.show()


patch_size = (199, 200)#37-19046,7-29430,50-15276
data = extract_patches_2d(jinface, patch_size,1)
print('printing data, means the 4900+ pacthes...')
print('data'+str(data))
print('data0'+str(data[0]))
#print('data1'+str(data[1]))
print(len(data))#4136 patches
print('整理后'+str(np.shape(data)))



print(jinface)
print('数据类型'+str(type(jinface)))
print('长度'+str(len(jinface)))#4136 patches
print('整理后'+str(np.shape(jinface)))
print('怎么导入多张照片,使上面这个值变成(多张图片张数,长,宽)')
