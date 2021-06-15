import cv2




def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                     (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def classify_hist_with_split(image1, image2, size=(256, 256)):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data



def similar(img1,img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    return  classify_hist_with_split(img1, img2)[0]

def main():

    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    print(classify_hist_with_split(img1,img2))


#https://zhuanlan.zhihu.com/p/93893211
if __name__ == '__main__':

    main()