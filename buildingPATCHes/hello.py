#coding=utf-8
# import numpy as np
# import cv2
# import scipy
# from matplotlib import pyplot as plt
#
# X = np.random.randint(25,50,(25,2))
# Y = np.random.randint(60,85,(25,2))
# Z = np.vstack((X,Y))
#
# # convert to np.float32
# Z = np.float32(Z)
#
# # define criteria and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret, label, center = cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#
# # Now separate the data, Note the flatten()
# A = Z[label.ravel()==0]
# B = Z[label.ravel()==1]
#
# # Plot the data
# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()
"""
=========================================
Image denoising using dictionary learning
=========================================

An example comparing the effect of reconstructing noisy fragments
of a raccoon face image using firstly online :ref:`DictionaryLearning` and
various transform methods.

The dictionary is fitted on the distorted left half of the image, and
subsequently used to reconstruct the right half. Note that even better
performance could be achieved by fitting to an undistorted (i.e.
noiseless) image, but here we start from the assumption that it is not
available.

A common practice for evaluating the results of image denoising is by looking
at the difference between the reconstruction and the original image. If the
reconstruction is perfect this will look like Gaussian noise.

It can be seen from the plots that the results of :ref:`omp` with two
non-zero coefficients is a bit less biased than when keeping only one
(the edges look less prominent). It is in addition closer from the ground
truth in Frobenius norm.

The result of :ref:`least_angle_regression` is much more strongly biased: the
difference is reminiscent of the local intensity value of the original image.

Thresholding is clearly not useful for denoising, but it is here to show that
it can produce a suggestive output with very high speed, and thus be useful
for other tasks such as object classification, where performance is not
necessarily related to visualisation.

"""
print(__doc__)

from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.image as mpimg # mpimg
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


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
face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
face /= 4.0
height, width = face.shape

# Distort the right half of the image
print('Distorting image...')
distorted = face.copy()
distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

# Extract all reference patches from the left half of the image
print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)#37-19046,7-29430,50-15276
#data = extract_patches_2d(distorted[:, :width // 2], patch_size)


data = mpimg.imread('7.jpg')
data = np.array(data)
data = data[:, :, 0]
data = data / 255.

# plt.imshow(jinface)
# plt.show()
data=data.reshape((1,7,7))


print("刚刚导出的patch"+str(np.shape(data)))
#print(data[1079])

data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

print('printing data, means the 4900+ pacthes...')
print(type(data))
print(len(data))#4136 patches
print('整理后'+str(np.shape(data)))
print('done in %.2fs.' % (time() - t0))

# #############################################################################
# Learn the dictionary from reference patches



print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
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

print('Extracting noisy patches... ')
t0 = time()
data = extract_patches_2d(distorted[:, width // 2:], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

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
