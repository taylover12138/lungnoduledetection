# -*-coding:utf-8-*-
import numpy as np

from matplotlib import pyplot as plt

working_path = r"D:\lungnoduledetection\result\output_subset1/"

imgs = np.load(working_path + 'masks_0008_0105.npy')
im= np.load(working_path+"images_0008_0105.npy")
# print(imgs.shape)
#
# plt.imshow(imgs, cmap='gray')  # 灰度图展示
# plt.imshow(im,cmap='gray')
# plt.show()
fig, ax = plt.subplots(1, 2, figsize=[8, 8])
ax[0].imshow(imgs, cmap='gray')
ax[1].imshow(im, cmap='gray')

plt.show()


