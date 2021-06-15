# -*- coding: utf-8 -*-
# from skimage.measure import compare_ssim
from scipy.misc import imread
import numpy as np
from skimage import measure
# import cv2
def similar(img1,img2):
    img1 = imread(img1)
    img2 = imread(img2)
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    # print(img1.shape)
    # print(img2.shape)
    return measure.compare_ssim(img1, img2, multichannel=True)

def  main():
    # 读取图片
    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"
    img1 = imread(img1)
    img2 = imread(img2)
    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    # print(img1.shape)
    # print(img2.shape)
    ssim =  measure.compare_ssim(img1, img2, multichannel = True)
    print(ssim)


#https://zhuanlan.zhihu.com/p/93893211
if __name__ == '__main__':
    main()