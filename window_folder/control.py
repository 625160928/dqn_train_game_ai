from ctypes import windll
import time

import win32api
import win32con
import win32gui

class Player():
    def __init__(self,player,window_name,time_sleep=0.1):
        self.time_sleep=time_sleep
        self.__handle = windll.user32.FindWindowW(None, window_name)

        self.__w2hd=win32gui.FindWindowEx(self.__handle, None, None, None)
        hwndChildList = []
        win32gui.EnumChildWindows(self.__handle, lambda hwnd, param: param.append(hwnd),  hwndChildList)

        for i in hwndChildList:
            if win32gui.GetClassName(i)=='Chrome_WidgetWin_0' :
                self.__handle=i
                break
            if win32gui.GetClassName(i)=='Chrome_RenderWidgetHostHWND' :
                self.__handle=i
                break

        if player=='red':
            self.__right=ord('D')
            self.__left=ord('A')
            self.__up=ord('W')
            self.__down=ord('S')
        elif player=='blue':
            self.__right=39
            self.__left=37
            self.__up=38
            self.__down=40
        else:
            print("没有这个角色")

    #horizontal 三个离散值，1右走，0不动，-1左走
    #jump 两个离散 0不跳，1跳
    def move(self,horizontal,jump):
        if horizontal==1:
            press_key=self.__right
        elif horizontal==-1:
            press_key=self.__left
        else:
            press_key=None

        win32gui.SetForegroundWindow(self.__w2hd)

        if press_key!=None:
            win32api.PostMessage(self.__handle, win32con.WM_KEYDOWN, press_key, 0)
        if jump==1:
            win32api.PostMessage(self.__handle, win32con.WM_KEYDOWN,self.__up, 0)

        time.sleep(self.time_sleep)

        win32gui.SetForegroundWindow(self.__w2hd)
        if press_key!=None:
            win32api.PostMessage(self.__handle, win32con.WM_KEYUP, press_key, 0)
        if jump==1:
            win32api.PostMessage(self.__handle, win32con.WM_KEYUP,self.__up, 0)






if __name__ == "__main__":

    window_name='FlashPlay'
    fire_man=Player(player='blue',window_name=window_name,time_sleep=0.2)
    for i in range(10):
        fire_man.move(-1,1)
    # handle = windll.user32.FindWindowW(None,window_name)
    # count=0
    # w2hd=win32gui.FindWindowEx(handle,None,None,None)
    # import time
    # start_time=time.time()
    # while(count<20):
    #     count+=1
    #     # win32gui.InSendMessage()
    #     # win32gui.ReplyMessage(0)
    #     win32gui.SetForegroundWindow(w2hd)
    #     win32api.keybd_event(68,0,0,0)
    #     time.sleep(1)
    #     win32gui.SetForegroundWindow(w2hd)
    #     win32api.keybd_event(68,0,win32con.KEYEVENTF_KEYUP,0)  #释放按键
