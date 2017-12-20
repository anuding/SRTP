#coding=utf-8
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.image as mpimg # mpimg
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
#
# jinface = mpimg.imread('jin.jpg')
# jinface = np.array(jinface)
# jinface = jinface[:, :, 0]
# jinface = jinface / 255.
# plt.imshow(jinface)
# plt.show()
#
#
# patch_size = (199, 200)#37-19046,7-29430,50-15276
# data = extract_patches_2d(jinface, patch_size,1)
# print('printing data, means the 4900+ pacthes...')
# print('data'+str(data))
# print('data0'+str(data[0]))
# #print('data1'+str(data[1]))
# print(len(data))#4136 patches
# print('整理后'+str(np.shape(data)))
#
#
#
# print(jinface)
# print('数据类型'+str(type(jinface)))
# print('长度'+str(len(jinface)))#4136 patches
# print('整理后'+str(np.shape(jinface)))
# print('怎么导入多张照片,使上面这个值变成(多张图片张数,长,宽)')




try:  # SciPy >= 0.16 have face in misc
    from scipy.misc import face# as face1
    face = mpimg.imread('jin.jpg')
    face = np.array(face)
    if face.ndim == 3:
       face = face[:, :, 0]
    plt.imshow(face)
    plt.show()
    #face = face(gray=True)
except ImportError:
    face = sp.face(gray=True)


# Convert from uint8 representation with values between 0 and 255 to
# a floating point representation with values between 0 and 1.
face = face / 255.

# downsample for higher speed
# face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
# face /= 4.0
height, width = face.shape

# Distort the right half of the image
print('Distorting image...')
distorted = face.copy()
# distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

# plt.imshow(distorted)
# plt.show()
print(width,height)
# Extract all reference patches from the left half of the image
print('Extracting reference patches...')
t0 = time()
patch_size = (195, 195)#37-19046,7-29430,50-15276
data = extract_patches_2d(distorted, patch_size,20)


jinface = mpimg.imread('195px_python.jpg')
jinface = np.array(jinface)
jinface = jinface[:, :, 0]
jinface = jinface / 255.
print("准备插入的patch"+str(np.shape(jinface)))
print(jinface)
# plt.imshow(jinface)
# plt.show()
data[19]=jinface
data[18]=jinface
data[17]=jinface#.copy()
# data[16]=jinface
# data[15]=jinface
# data[14]=jinface
# data[13]=jinface
# data[12]=jinface
data[11]=jinface
data[10]=jinface
data[9]=jinface
data[8]=jinface
data[7]=jinface
data[6]=jinface
data[5]=jinface
data[4]=jinface
data[3]=jinface
data[2]=jinface
data[1]=jinface
data[0]=jinface
# plt.imshow(data[13])
# plt.show()
# data = mpimg.imread('7.jpg')
# data = np.array(data)
# data = data[:, :, 0]
# data = data / 255.
#
# # plt.imshow(jinface)
# # plt.show()
# data=data.reshape((1,7,7))


print("刚刚导出的patch"+str(np.shape(data)))
#print(data[1079])
# print(data)
# print(data[0])
data = data.reshape(data.shape[0], -1)

# data -= np.mean(data, axis=0)
# print(data[0])
# data /= np.std(data, axis=0)
# print(data[0])

print('printing data, means the 4900+ pacthes...')
print(type(data))
print(len(data))#4136 patches
print('整理后'+str(np.shape(data)))
print('done in %.2fs.' % (time() - t0))


print(data)
# for i in range(len(data)):
#     sample = data[i]
#     for j in range(len(sample)):
#         if np.isnan(sample[j]):
#             sample[j] = 0
        # #############################################################################
# Learn the dictionary from reference patches



print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=3, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

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


# #############################################################################
# Display the distorted image

def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

#show_with_diff(distorted, face, 'Distorted image')

# #############################################################################
# Extract noisy patches and reconstruct them using the dictionary

# print('Extracting noisy patches... ')
# t0 = time()
# data = extract_patches_2d(distorted[:, width // 2:], patch_size)
# data = data.reshape(data.shape[0], -1)
# intercept = np.mean(data, axis=0)
# data -= intercept
# print('done in %.2fs.' % (time() - t0))

transform_algorithms = [
    # ('Orthogonal Matching Pursuit\n1 atom', 'omp',
    #  {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 1})]
    # ('Least-angle regression\n5 atoms', 'lars',
    #  {'transform_n_nonzero_coefs': 5}),
    # ('Thresholding\n alpha=0.1', 'threshold',
    #  {'transform_alpha': .1})]

reconstructions = {}
# for title, transform_algorithm, kwargs in transform_algorithms:
#     print(title + '...')
#     reconstructions[title] = face.copy()
#     t0 = time()
#     dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
#     code = dico.transform(data)
#     patches = np.dot(code, V)
#
#     patches += intercept
#     patches = patches.reshape(len(data), *patch_size)
#     if transform_algorithm == 'threshold':
#         patches -= patches.min()
#         patches /= patches.max()
#     reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(
#         patches, (height, width // 2))
#     dt = time() - t0
#     print('done in %.2fs.' % dt)
#     show_with_diff(reconstructions[title], face,
#                    title + ' (time: %.1fs)' % dt)

plt.show()