from ctypes import windll
from ctypes.wintypes import HWND
import string
import time

import win32api
import win32con
import sys
import win32gui

PostMessageW = windll.user32.PostMessageW
MapVirtualKeyW = windll.user32.MapVirtualKeyW
VkKeyScanA = windll.user32.VkKeyScanA

WM_KEYDOWN = 0x100
WM_KEYUP = 0x101

# https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
VkCode = {
    "back":  0x08,
    "tab":  0x09,
    "return":  0x0D,
    "shift":  0x10,
    "control":  0x11,
    "menu":  0x12,
    "pause":  0x13,
    "capital":  0x14,
    "escape":  0x1B,
    "space":  0x20,
    "end":  0x23,
    "home":  0x24,
    "left":  0x25,
    "up":  0x26,
    "right":  0x27,
    "down":  0x28,
    "print":  0x2A,
    "snapshot":  0x2C,
    "insert":  0x2D,
    "delete":  0x2E,
    "lwin":  0x5B,
    "rwin":  0x5C,
    "numpad0":  0x60,
    "numpad1":  0x61,
    "numpad2":  0x62,
    "numpad3":  0x63,
    "numpad4":  0x64,
    "numpad5":  0x65,
    "numpad6":  0x66,
    "numpad7":  0x67,
    "numpad8":  0x68,
    "numpad9":  0x69,
    "multiply":  0x6A,
    "add":  0x6B,
    "separator":  0x6C,
    "subtract":  0x6D,
    "decimal":  0x6E,
    "divide":  0x6F,
    "f1":  0x70,
    "f2":  0x71,
    "f3":  0x72,
    "f4":  0x73,
    "f5":  0x74,
    "f6":  0x75,
    "f7":  0x76,
    "f8":  0x77,
    "f9":  0x78,
    "f10":  0x79,
    "f11":  0x7A,
    "f12":  0x7B,
    "numlock":  0x90,
    "scroll":  0x91,
    "lshift":  0xA0,
    "rshift":  0xA1,
    "lcontrol":  0xA2,
    "rcontrol":  0xA3,
    "lmenu":  0xA4,
    "rmenu":  0XA5
}


def get_virtual_keycode(key: str):
    """根据按键名获取虚拟按键码

    Args:
        key (str): 按键名

    Returns:
        int: 虚拟按键码
    """
    if len(key) == 1 and key in string.printable:
        # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-vkkeyscana
        return VkKeyScanA(ord(key)) & 0xff
    else:
        return VkCode[key]


def key_down(handle: HWND, key: str):
    """按下指定按键

    Args:
        handle (HWND): 窗口句柄
        key (str): 按键名
    """
    vk_code = get_virtual_keycode(key)
    scan_code = MapVirtualKeyW(vk_code, 0)
    # https://docs.microsoft.com/en-us/windows/win32/inputdev/wm-keydown
    wparam = vk_code
    lparam = (scan_code << 16) | 1
    PostMessageW(handle, WM_KEYDOWN, wparam, lparam)


def key_up(handle: HWND, key: str):
    """放开指定按键

    Args:
        handle (HWND): 窗口句柄
        key (str): 按键名
    """
    vk_code = get_virtual_keycode(key)
    scan_code = MapVirtualKeyW(vk_code, 0)
    # https://docs.microsoft.com/en-us/windows/win32/inputdev/wm-keyup
    wparam = vk_code
    lparam = (scan_code << 16) | 0XC0000001
    PostMessageW(handle, WM_KEYUP, wparam, lparam)

#parent为父窗口句柄id
def get_child_windows(handle):
    '''
    获得parent的所有子窗口句柄
     返回子窗口句柄列表
     '''
    if not handle:
        return
    hwndChildList = []
    win32gui.EnumChildWindows(handle, lambda hwnd, param: param.append(hwnd), hwndChildList)
    return hwndChildList



if __name__ == "__main__":
    # 需要和目标窗口同一权限，游戏窗口通常是管理员权限

    # if not windll.shell32.IsUserAnAdmin():
    #     # 不是管理员就提权
    #     windll.shell32.ShellExecuteW(
    #         None, "runas", sys.executable, __file__, None, 1)



    window_name='FlashPlay'
    # window_name='森林冰火人大冒险2选关版小游戏,在线玩,4399小游戏 - 视频播放器'

    handle = windll.user32.FindWindowW(None,window_name)
    print(handle)
    son=get_child_windows(handle)
    print(son)
    print('useless',son[0],handle)
    print('class name ',win32gui.GetClassName(son[0]),win32gui.GetClassName(handle))
    print('useful',son[1],son[2])
    print('class name ',win32gui.GetClassName(son[1]),win32gui.GetClassName(son[2]))

#
    # if handle!=0:
    #     # 控制角色向前移动两秒
    #     # key_down(handle, 'd')
    #     # time.sleep(2)
    #     # key_up(handle, 'd')
    #
    #     # win32api.keybd_event(13,0,0,0)
    #     # win32api.keybd_event(13,0,win32con.KEYEVENTF_KEYUP,0)
    #     # win32gui.SendMessage(handle, WM_KEYDOWN, 'D', 0)
    #     count=0
    #     w2hd=win32gui.FindWindowEx(handle,None,None,None)
    #     import time
    #     start_time=time.time()
    #     while(count<1000):
    #         count+=1
    #         # win32gui.PostMessage(handle, win32con.WM_CLOSE, 0, 0)
    #
    #         # win32gui.InSendMessage()
    #         # win32gui.ReplyMessage(0)
    #         # win32gui.SetForegroundWindow(w2hd)
    #         # win32api.keybd_event(68,0,0,0)
    #         # time.sleep(1)
    #         win32gui.SetForegroundWindow(w2hd)
    #         # win32api.keybd_event(68,0,win32con.KEYEVENTF_KEYUP,0)  #释放按键
    #
    #
    #         # a1 = win32gui.PostMessage(handle, win32con.WM_KEYDOWN, win32con.VK_RIGHT, 10)
    #         # a2 = win32api.SendMessage(handle, win32con.WM_NCMBUTTONDOWN, win32con.VK_RIGHT, 0)
    #         # win32api.PostMessage(handle, win32con.WM_KEYDOWN, ord('D'), 0)
    #         # win32api.PostMessage(handle, win32con.WM_CHAR, ord('D'), 0)
    #         # key_down(handle,'d')
    #         # key_down(handle,'D')
    #         # for i in range(len(son)):
    #         #
    #         #     # key_down(i,'d')
    #         #     # key_down(i,'D')
    #         #     # a1=win32gui.PostMessage(i, win32con.WM_KEYDOWN,  win32con.VK_RIGHT, 10)
    #         #     # a2=win32api.SendMessage(i, win32con.WM_NCMBUTTONDOWN, win32con.VK_RIGHT, 0)
    #         #     win32api.PostMessage(son[2], win32con.WM_KEYDOWN, ord('D'), 0)
    #         #     win32api.PostMessage(son[2], win32con.WM_CHAR, ord('D'), 0)
    #         #     a=1
    #         #     # print(a1)
    #         #     # print(a2)