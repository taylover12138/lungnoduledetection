# -*-coding:utf-8-*-
# -*- coding:utf-8 -*-
'''
此脚本用于数据科学碗中的lung 2017基本流程
'''
import SimpleITK as sitk
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

# numpyImage[numpyImage > -600] = 1
# numpyImage[numpyImage <= -600] = 0
#归一化
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg
def get_segmented_lungs(im, plot=False):
    '''
 	该功能从给定的2D切片分割肺部。
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: 转换成二进制图像。
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: 移除连接到图像边框的斑点。
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: 给图片贴上标签。
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4:保持标签上有两个最大的区域。
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: 使用半径为2的圆盘进行侵蚀操作。这个手术是分离附在血管上的肺结节。
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: 使用半径为10的圆盘进行闭合操作。这个手术是为了让结节附着在肺壁上。
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: 填充肺部二元面罩内的小孔。
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # binary=lumTrans(binary)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: 在输入图像上叠加二值遮罩。
    '''
    sum = 0
    for r in regionprops(label_image):
         sum = sum + r.area
    proportion = sum / (512 * 512)
    #im = (255 / 1800 * im + 1200 * 255 / 1800)#缩放至[0-255]
    im=lumTrans(im)
    get_high_vals = binary == 0
    im[get_high_vals] =170#空白区域设置为170
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    plt.show()

    return im,proportion


if __name__ == '__main__':
    filename = r'D:\lungnoduledetection\dataset\subset2\1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286.mhd'
    itkimage = sitk.ReadImage(filename)  # 读取.mhd文件
    numpyImage = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取
    data = numpyImage[50]
    LP = []  # lung proportion
    resample_im = []
    plt.figure(50)
    plt.imshow(data, cmap='gray')
    im,proportion= get_segmented_lungs(data, plot=False)

    LP.append(proportion)
    resample_im.append(im)
    plt.figure(200)
    plt.imshow(im, cmap='gray')
    plt.show()

