import cv2
import numpy as np

def phash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash



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


def phash_sim(img1,img2,log=False):
    hash1 = phash(img1)
    hash3 = phash(img2)
    if log:
        print('img1的phash值', hash1)
        print('img2的phash值', hash3)
    camphash1 = campHash(hash1, hash3)
    return camphash1/len(hash1)

def similar(img1,img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    return phash_sim(img1,img2)

def main():
    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    print("phash差异哈希相似度：", phash_sim(img1,img2)*100)


#https://zhuanlan.zhihu.com/p/93893211
if __name__ == "__main__":
    img0 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    # random_sim(img0)
    main()