# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg # mpimg
# import numpy as np
# import scipy as sp
# from scipy.misc import face
#
#
# import matplotlib.cbook as cbook
#
# #image_file = cbook.get_sample_data('ada.png')
# image = mpimg.imread("f1.jpg")
#
# plt.imshow(image)
# plt.axis('off')  # clear x- and y-axes
# plt.show()
# -*-coding:gbk-*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('jin.jpg')
img = np.array(img)
if img.ndim == 3:
    img = img[:,:,0]
plt.subplot(221); plt.imshow(img)
plt.subplot(222); plt.imshow(img, cmap ='gray')
plt.subplot(223); plt.imshow(img, cmap = plt.cm.gray)
plt.subplot(224); plt.imshow(img, cmap = plt.cm.gray_r)
plt.savefig("gray.png")
plt.show()
