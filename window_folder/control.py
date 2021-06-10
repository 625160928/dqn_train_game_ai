from ctypes import windll
from ctypes.wintypes import HWND
import string
import time

import win32api
import win32con
import win32gui

class Player():
    def __init__(self,player,window_name):
        self.time_sleep=0.1
        handle = windll.user32.FindWindowW(None,window_name)
        self.w2hd=win32gui.FindWindowEx(handle,None,None,None)

        if player=='red':
            self.right=68
            self.left=65
            self.up=87
            self.down=83
        elif player=='blue':
            self.right=39
            self.left=37
            self.up=38
            self.down=40
        else:
            print("没有这个角色")

    #horizontal 三个离散值，1右走，0不动，-1左走
    #jump 两个离散 0不跳，1跳
    def move(self,horizontal,jump):
        if horizontal==1:
            press_key=self.right
        elif horizontal==-1:
            press_key=self.left
        else:
            press_key=None
        win32gui.SetForegroundWindow(self.w2hd)
        if press_key!=None:
            win32api.keybd_event(press_key,0,0,0)
        if jump==1:
            win32api.keybd_event(self.up,0,0,0)
        time.sleep(self.time_sleep)
        if press_key!=None:
            win32api.keybd_event(press_key,0,win32con.KEYEVENTF_KEYUP,0)
        if jump==1:
            win32api.keybd_event(self.up,0,win32con.KEYEVENTF_KEYUP,0)





if __name__ == "__main__":

    window_name='FlashPlay'
    handle = windll.user32.FindWindowW(None,window_name)
    count=0
    w2hd=win32gui.FindWindowEx(handle,None,None,None)
    import time
    start_time=time.time()
    while(count<20):
        count+=1
        # win32gui.InSendMessage()
        # win32gui.ReplyMessage(0)
        win32gui.SetForegroundWindow(w2hd)
        win32api.keybd_event(68,0,0,0)
        time.sleep(1)
        win32gui.SetForegroundWindow(w2hd)
        win32api.keybd_event(68,0,win32con.KEYEVENTF_KEYUP,0)  #释放按键
