# -*-coding:utf-8-*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
newFilePath1='D:\lungnoduledetection\result\preprocess_result_pathsubset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260_mask.npy'
aa=np.array([1,2,3,5])
print(aa[0:2])
newVolume1 = np.load(newFilePath1)
print(len(newVolume1))
for i in range(len(newVolume1)):
    if i%5==0:
        print(i)
        data=newVolume1[i]
        plt.figure(i)
        plt.imshow(data,cmap='gray')
plt.show()