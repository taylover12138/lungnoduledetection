# -*-coding:utf-8-*-
# -*- coding:utf-8 -*-
'''
此脚本用于数据科学碗中的lung 2017基本流程
'''
import SimpleITK as sitk
from skimage.morphology import disk,binary_erosion,binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from glob import glob
try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

# numpyImage[numpyImage > -600] = 1
# numpyImage[numpyImage <= -600] = 0
#归一化到[255]
def lumTrans(img):
    lungwin = np.array([-1000.,600.])
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

    return im







if __name__ == '__main__':
    # filename = r'D:\lungnoduledetection\dataset\subset1\1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mhd'
    luna_path = r"D:\lungnoduledetection\dataset/"
    outluna_path = r"D:\lungnoduledetection\result2/"
    luna_subset_path = luna_path + "subset1/"
    output_path = outluna_path + "output_subset1/"
    file_list = glob(luna_subset_path + "*.mhd")
    file_list_path=[]
    for i in range(len(file_list)):
        file_list_path.append(file_list[i][0:-4])


    # Helper function to get rows in data frame associated
    # with each file
    def get_filename(file_list, case):
        for f in file_list:
            if case in f:
                return (f)
    # The locations of the nodes
    df_node = pd.read_csv("D:\\lungnoduledetection\\dataset\\CSVFILES\\annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list_path, file_name))
    df_node = df_node.dropna()

    # Looping over the image files
    #
    for fcount, img_file in enumerate(tqdm(file_list_path)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those

            img_file = img_file + ".mhd"
            itkimage = sitk.ReadImage(img_file)  # 读取.mhd文件
            numpyImage = sitk.GetArrayFromImage(itkimage)# 获取数据，自动从同名的.raw文件读取



            num_z, height, width = numpyImage.shape  # heightXwidth constitute the transverse plane
            origin = np.array(itkimage.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itkimage.GetSpacing())  # spacing of voxels in world coor. (mm)
            for node_idx, cur_row in mini_df.iterrows():
                        node_x = cur_row["coordX"]
                        node_y = cur_row["coordY"]
                        node_z = cur_row["coordZ"]
                        diam = cur_row["diameter_mm"]


                        imgs = np.ndarray([height, width], dtype=np.float32)

                        center = np.array([node_x, node_y, node_z])  # nodule center
                        v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)

                        for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                          int(v_center[2]) + 2).clip(0,
                                                                                     num_z - 1)):  # clip prevents going out of bounds in Z
                            # print('i=')
                            # print(i)
                            # print('i_z=')
                            i_z = i_z + 1
                            # print(i_z)


                            imgs = numpyImage[i_z]

                            # print(imgs.shape)
                            # for i in range(imgs.shape[0]):
                            #     data=imgs[i]
                            #     # print(data.shape)
                            #
                            im= get_segmented_lungs(imgs, plot=False)
                            print(im.shape)
                            np.save(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount,node_idx)), im)
                            #     plt.figure(200)
                            #     plt.imshow(im, cmap='gray')
                            #     plt.show()