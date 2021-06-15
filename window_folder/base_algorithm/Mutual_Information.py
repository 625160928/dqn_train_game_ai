from sklearn import metrics as mr
from scipy.misc import imread
import numpy as np

def similar(img1,img2):
    img1 = imread(img1)
    img2 = imread(img2)

    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))

    img1 = np.reshape(img1, -1)
    img2 = np.reshape(img2, -1)
    # print(img2.shape)
    # print(img1.shape)
    return  mr.mutual_info_score(img1, img2)

def main():

    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"

    img1 = imread(img1)
    img2 = imread(img2)

    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))

    img1 = np.reshape(img1, -1)
    img2 = np.reshape(img2, -1)
    # print(img2.shape)
    # print(img1.shape)
    mutual_infor = mr.mutual_info_score(img1, img2)

    print(mutual_infor)


#https://zhuanlan.zhihu.com/p/93893211
if __name__ == '__main__':
    main()