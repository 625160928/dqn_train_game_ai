import cv2
import numpy as np

dhash_N=16

# 差异值哈希算法
def dhash(image):
    n=dhash_N
    # 将图片转化为8*8
    image = cv2.resize(image, (n+1, n), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(n):
        for j in range(n):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, n*n, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result


def dhash_sim(img1,img2,log=False):
    hash1 = dhash(img1)
    hash3 = dhash(img2)
    if log:
        print('img1的dhash值', hash1)
        print('img2的dhash值', hash3)
    camphash1 = campHash(hash1, hash3)
    return camphash1/len(hash1)


# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def similar(img1,img2):

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    return dhash_sim(img1,img2)

def main():
    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    # print("ahash均值哈希相似度：", ahash_sim(img1,img2)*100)
    print("dhash差异哈希相似度：", dhash_sim(img1,img2)*100)
    # print("phash差异哈希相似度：", phash_sim(img1,img2)*100)

#计算dhash相似度
if __name__ == "__main__":
    img0 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    # random_sim(img0)
    main()