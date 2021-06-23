import matplotlib.pyplot as plt
import numpy as np

from  control import Player
from get_win import window_capture
from window_folder import Game_State

class game_env():
    def __init__(self,window_name,player_name,resize_w,resize_h,reccognise):
        self.__player=Player(player_name,window_name)
        self.__env=window_capture(window_name,resize_h,resize_w)
        self.__game_state=Game_State.game_state(window_name,reccognise,resize_h,resize_w)
        #识别区域
        self.__reccognise_x=reccognise[0]
        self.__reccognise_y=reccognise[1]
        self.__reccognise_h=reccognise[2]
        self.__reccognise_w=reccognise[3]

    def act(self,horizontal,jump):
        self.__player.move(horizontal,jump)
        state=self.__env.capture()
        arr=np.array(state)
        # print(type(arr))
        return arr

    def set_level(self,level):
        return

    def get_game_state(self):
        game_state_pic=self.__env.capture_part(self.__reccognise_x,self.__reccognise_y,self.__reccognise_h,self.__reccognise_w)
        game_state_arr=np.array(game_state_pic)

        return game_state_arr

    def restart(self):
        a=1




def print_game_state(state):
    arr=np.array(state)
    print('np.array([')
    for i in range(len(arr)):
        print('[',end='')
        for j in range(len(arr[i])):
            print('[',arr[i][j][0],',',arr[i][j][1],',',arr[i][j][2],',',arr[i][j][3],'],',end='')
        print('],')
    print('])')

if __name__ == "__main__":
    window_name='FlashPlay'
    player_name='red'
    reccognise_x=46
    reccognise_y=50
    reccognise_h=15
    reccognise_w=15
    resize_h=300
    resize_w=400
    reccognise=[reccognise_x,reccognise_y,reccognise_h,reccognise_w]
    game_env0=game_env(window_name,player_name,resize_w,resize_h,reccognise)

    # while(1):
    #     state=game_env0.get_game_state()
    #
    #     print(state)

    state=game_env0.get_game_state()
    # print(state)
    # print("------------------")
    print(print_game_state(state))


    # print(np.array(state))

    # # print(state)
    # plt.imshow(state)
    # # plt.pause(0.01)
    # plt.show()