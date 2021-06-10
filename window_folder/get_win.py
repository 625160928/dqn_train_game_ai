from ctypes import windll, byref, c_ubyte
from ctypes.wintypes import RECT, HWND

import matplotlib.pyplot as plt
import numpy as np
import cv2

class window_capture():
    def __init__(self,window_name,reshape_height,reshape_width):
        self.handle = windll.user32.FindWindowW(None, window_name)
        self.reshape_height=reshape_height
        self.reshape_width=reshape_width

    def capture(self):
        """窗口客户区截图

        Args:
            handle (HWND): 要截图的窗口句柄

        Returns:
            numpy.ndarray: 截图数据
        """
        # 获取窗口客户区的大小
        r = RECT()
        windll.user32.SetProcessDPIAware()
        windll.user32.GetClientRect(self.handle, byref(r))
        width, height = r.right, r.bottom
        # 开始截图
        dc = windll.user32.GetDC(self.handle)
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
        windll.user32.ReleaseDC(self.handle, dc)
        # 返回截图数据为numpy.ndarray
        return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        # return np.frombuffer(buffer, dtype=np.uint8).reshape(self.reshape_height, self.reshape_width, 4)

if __name__ == "__main__":
    window_name='FlashPlay'
    win_c=window_capture(window_name,600,800)
    count=0
    while count<300:
        image =  win_c.capture()
        print(image.shape,image[950][1250])
        # print(image)
        # cv2.imshow("Capture Test", image)
        # cv2.waitKey()
        plt.imshow(image)
        plt.pause(0.01)
        count+=1