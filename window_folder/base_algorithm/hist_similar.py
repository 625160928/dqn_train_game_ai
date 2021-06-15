# 将图片转化为RGB
from PIL import Image
regular_size=64

def make_regalur_image(img, size=(regular_size, regular_size)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image


# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 计算相似度
def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim

def similar(img1,img2):
    image1 = Image.open(img1)
    image1 = make_regalur_image(image1)
    image2 = Image.open(img2)
    image2 = make_regalur_image(image2)
    return calc_similar(image1, image2)


#计算直方图相似度
if __name__ == '__main__':
    img1 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/1.JPG"
    img2 = "F:/Python_project_intrestest/photo_simular/image_test/TEST4/2.JPG"

    image1 = Image.open(img1)
    image1 = make_regalur_image(image1)
    image2 = Image.open(img2)
    image2 = make_regalur_image(image2)
    print("图片间的相似度为", calc_similar(image1, image2))
