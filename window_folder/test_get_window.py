import matplotlib.pyplot as plt
from win32gui import FindWindow
import win32gui
import win32api,win32con
# use 'pip install pywin32' to install
import win32api, win32con, win32gui
from PIL import Image, ImageGrab
#对后台窗口截图
import win32gui, win32ui, win32con
from ctypes import windll
from PIL import Image
import cv2
import numpy



def get_window_pos(name):
    name = name
    handle = win32gui.FindWindow(0, name)
    # 获取窗口句柄
    if handle == 0:
        return None
    else:
        # 返回坐标值和handle
        return win32gui.GetWindowRect(handle), handle


def fetch_image(name):
    (x1, y1, x2, y2), handle = get_window_pos(name)
    print((x1, y1, x2, y2), handle)
    # 发送还原最小化窗口的信息
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # 设为高亮
    win32gui.SetForegroundWindow(handle)
    # 截图
    grab_image = ImageGrab.grab((x1, y1, x2, y2))

    return grab_image

#parent为父窗口句柄id
def get_child_windows(parent):
    '''
    获得parent的所有子窗口句柄
     返回子窗口句柄列表
     '''
    if not parent:
        return
    hwndChildList = []
    win32gui.EnumChildWindows(parent, lambda hwnd, param: param.append(hwnd),  hwndChildList)
    return hwndChildList

def test():
    window_name='FlashPlay'

    handler = FindWindow(None, window_name)
    # jbid=handler
    print('handler ',handler)

    son=get_child_windows(handler)
    print('son ',son)

    jbid=handler
    print('jbid ',jbid)
    left, top, right, bottom = win32gui.GetWindowRect(handler)
    print('left, top, right, bottom',left, top, right, bottom)

    #获取标题
    title = win32gui.GetWindowText(jbid)
    print('title ',title)

    #获取类名
    clsname = win32gui.GetClassName(jbid)
    print('class name ',clsname)

    # #根据横纵坐标定位光标
    # win32api.SetCursorPos([(int)((left+right)/2), (int)((top+bottom)/2)])
    #
    # #根据句柄将窗口放在最前
    # win32gui.SetForegroundWindow(jbid)
    #
    # #给光标定位的位置进行右击操作
    # win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP | win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    #
    # #给光标定位的位置进行单击操作（若想进行双击操作，可以延时几毫秒再点击一次）
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    img=fetch_image(window_name)
    plt.imshow(img)
    plt.show()

#获取窗口名字
def winEnumHandler(hwnd,non):
    if win32gui.IsWindowVisible(hwnd):
        name=win32gui.GetWindowText(hwnd)
        if name!='':
            print(hex(hwnd),name )

if __name__ == '__main__':

    win32gui.EnumWindows(winEnumHandler,None)
    # get_win()
    # test()



