# -*- coding: utf-8 -*-
# !/usr/bin/env python
# 余弦相似度计算
from PIL import Image
from numpy import average, dot, linalg
# 对图片进行统一化处理
def get_thum(image, size=(15, 15), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image
# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

def similar(img1,img2):
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    return image_similarity_vectors_via_numpy(image1, image2)


def main():

    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    cosin = image_similarity_vectors_via_numpy(image1, image2)
    print('图片余弦相似度', cosin)

#https://zhuanlan.zhihu.com/p/93893211
if __name__ == "__main__":
    main()