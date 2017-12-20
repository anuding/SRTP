#coding=utf-8
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.image as mpimg # mpimg
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

imgpath='res/'

#导入人为设置的第一个patch的原图像,名为face,是200*200的金馆长,patch大小后文设为195*195
try:
    from scipy.misc import face# as face1
    face = mpimg.imread(imgpath+'200px_jin.jpg')
    face = np.array(face)
    if face.ndim == 3:
       face = face[:, :, 0]                             #二值化金馆长
    # plt.imshow(face)
    # plt.show()                                        #这里可以取消注释使二值化的结果显示
except ImportError:
    face = sp.face(gray=True)



face = face / 255.                                      #将图像矩阵中的值范围归到[0,1]
height, width = face.shape


print('Extracting  patches...')
t0 = time()#37-19046,7-29430,50-15276  测试数据
patch_size = (195, 195)                                  #一个字的大小
data = extract_patches_2d(face, patch_size,20)           #从原始数据face里生成20个195*195的字(patch)

secondpatch = mpimg.imread(imgpath+'195px_python.jpg')   #读取第二种patch的图像,名为secondpatch,是195*195的python图标
secondpatch = np.array(secondpatch)
secondpatch =secondpatch [:, :, 0]
secondpatch = secondpatch / 255.                    #预处理patch
print("准备插入的patch"+str(np.shape(secondpatch)))  #显示第二种patch的形状
#print(secondpatch)                                  #显示第二种patch的数据


#用第二种patch替换一部分第一种patch
data[19]=secondpatch
data[18]=secondpatch
data[17]=secondpatch#.copy()
# data[16]=secondpatch
# data[15]=secondpatch
# data[14]=secondpatch
# data[13]=secondpatch
# data[12]=secondpatch
data[11]=secondpatch
data[10]=secondpatch
data[9]=secondpatch
data[8]=secondpatch
data[7]=secondpatch
data[6]=secondpatch
data[5]=secondpatch
data[4]=secondpatch
data[3]=secondpatch
data[2]=secondpatch
data[1]=secondpatch
data[0]=secondpatch

print("刚刚导出的patch"+str(np.shape(data)))         #至此,已形成patch(20L, 195L, 195L),即有20个字,每个字长宽均为195px
data = data.reshape(data.shape[0], -1)              #格式改成字典学习函数能接受的(20L,38025L)

# data -= np.mean(data, axis=0)
# print(data[0])
# data /= np.std(data, axis=0)
# print(data[0])                                    #源程序的标准化代码,如果不注释掉就会除0产生无限大


print('整理后'+str(np.shape(data)))                 #显示data改完格式后的形状
print('done in %.2fs.' % (time() - t0))


print('Learning the dictionary...')
t0 = time()
#开始字典学习,n_components表示选多少个出来,可以理解为前几个最具代表性的帧,n_iter代表迭代次数
dico = MiniBatchDictionaryLearning(n_components=10, alpha=1, n_iter=1000)   #dico是字典学习类的一个对象
V = dico.fit(data).components_                     #调用训练函数fit,使用前面做的data作为字典,V即为最终获得的关键字
dt = time() - t0
print('done in %.2fs.' % dt)



#以下是显示最终的图片的代码
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)



plt.show()