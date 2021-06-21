from ctypes import windll, byref, c_ubyte
from ctypes.wintypes import RECT, HWND

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class window_capture():
    def __init__(self,window_name,reshape_height,reshape_width):
        self.__handle = windll.user32.FindWindowW(None, window_name)
        self.__reshape_height=reshape_height
        self.__reshape_width=reshape_width

    def capture(self):
        """窗口客户区截图

        Args:
            handle (HWND): 要截图的窗口句柄
        a
        Returns:
            numpy.ndarray: 截图数据
        """
        # 获取窗口客户区的大小
        r = RECT()
        windll.user32.SetProcessDPIAware()
        windll.user32.GetClientRect(self.__handle, byref(r))
        width, height = r.right, r.bottom
        # 开始截图
        dc = windll.user32.GetDC(self.__handle)
        cdc = windll.gdi32.CreateCompatibleDC(dc)
        bitmap = windll.gdi32.CreateCompatibleBitmap(dc, width, height)
        windll.gdi32.SelectObject(cdc, bitmap)
        windll.gdi32.BitBlt(cdc, 0, 0, width, height, dc, 0, 0, 0x00CC0020)
        # 截图是BGRA排列，因此总元素个数需要乘以4
        total_bytes = width*height*4
        buffer = bytearray(total_bytes)
        byte_array = c_ubyte*total_bytes
        windll.gdi32.GetBitmapBits(bitmap, total_bytes, byte_array.from_buffer(buffer))
        windll.gdi32.DeleteObject(bitmap)
        windll.gdi32.DeleteObject(cdc)
        windll.user32.ReleaseDC(self.__handle, dc)

        img_arr=np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        image_resize = Image.fromarray(img_arr).resize((self.__reshape_width,self.__reshape_height))

        return image_resize

    def capture_part(self,start_x,start_y,height,width):
        """窗口客户区截图

        Args:
            handle (HWND): 要截图的窗口句柄

        Returns:
            numpy.ndarray: 截图数据
        """
        # 获取窗口客户区的大小
        # r = RECT()
        windll.user32.SetProcessDPIAware()
        # windll.user32.GetClientRect(self.__handle, byref(r))
        # width, height = r.right, r.bottom
        # 开始截图
        dc = windll.user32.GetDC(self.__handle)
        cdc = windll.gdi32.CreateCompatibleDC(dc)
        bitmap = windll.gdi32.CreateCompatibleBitmap(dc, width, height)
        windll.gdi32.SelectObject(cdc, bitmap)
        windll.gdi32.BitBlt(cdc, 0,0, width, height, dc, start_x, start_y, 0x00CC0020)
        # 截图是BGRA排列，因此总元素个数需要乘以4
        total_bytes = width*height*4
        buffer = bytearray(total_bytes)
        byte_array = c_ubyte*total_bytes
        windll.gdi32.GetBitmapBits(bitmap, total_bytes, byte_array.from_buffer(buffer))
        windll.gdi32.DeleteObject(bitmap)
        windll.gdi32.DeleteObject(cdc)
        windll.user32.ReleaseDC(self.__handle, dc)

        img_arr=np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        # image_resize = Image.fromarray(img_arr).resize((self.__reshape_width,self.__reshape_height))
        # plt.show(img_arr)
        # plt.show()
        return img_arr


if __name__ == "__main__":
    window_name='FlashPlay'
    win_c=window_capture(window_name,reshape_height=300,reshape_width=400)
    while(1):
        plt.imshow(win_c.capture_part(20,40,70,265))
        plt.pause(0.01)
    # count=0
    # while(count<1):
    #     count+=1
    #     photo=win_c.capture()
    #     print(photo)
    #     plt.imshow(photo)
    #     plt.pause(0.01)

