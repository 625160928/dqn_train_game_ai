import cv2
import matplotlib.pyplot as plt
import numpy as np
from get_win import window_capture
from base_algorithm import cos_similar
from PIL import Image
import  base_algorithm
from base_algorithm import hist_similar
from base_algorithm import three_hist


SELECT=0
FAIL=-1
MENU=1
GAMEING=2
WIN=3
PROCESSING=4


class game_state():
    def __init__(self,window_name,reccognise,resize_h,resize_w):
        self.__env=window_capture(window_name,resize_h,resize_w)
        self.fail=np.array([
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],[ 198 , 104 , 81 , 255 ],],
            [[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],],
            [[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],[ 198 , 105 , 82 , 255 ],],
            [[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],[ 198 , 105 , 83 , 255 ],],
        ])
        self.select=np.array([
            [[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],],
            [[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],],
            [[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],],
            [[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],],
            [[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],],
            [[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 217 , 157 , 37 , 255 ],[ 217 , 157 , 37 , 255 ],[ 217 , 157 , 37 , 255 ],[ 215 , 155 , 37 , 255 ],[ 215 , 155 , 37 , 255 ],[ 211 , 152 , 39 , 255 ],],
            [[ 206 , 147 , 41 , 255 ],[ 204 , 145 , 42 , 255 ],[ 201 , 142 , 43 , 255 ],[ 201 , 142 , 43 , 255 ],[ 198 , 140 , 44 , 255 ],[ 196 , 137 , 44 , 255 ],[ 193 , 135 , 46 , 255 ],[ 193 , 135 , 46 , 255 ],[ 193 , 135 , 46 , 255 ],[ 191 , 132 , 47 , 255 ],[ 188 , 130 , 48 , 255 ],[ 188 , 130 , 48 , 255 ],[ 185 , 127 , 49 , 255 ],[ 182 , 125 , 50 , 255 ],[ 180 , 122 , 51 , 255 ],],
            [[ 181 , 122 , 51 , 255 ],[ 176 , 117 , 53 , 255 ],[ 172 , 115 , 54 , 255 ],[ 167 , 110 , 56 , 255 ],[ 163 , 105 , 58 , 255 ],[ 159 , 102 , 60 , 255 ],[ 157 , 100 , 60 , 255 ],[ 154 , 97 , 61 , 255 ],[ 154 , 97 , 61 , 255 ],[ 152 , 95 , 63 , 255 ],[ 149 , 92 , 63 , 255 ],[ 149 , 92 , 63 , 255 ],[ 146 , 90 , 65 , 255 ],[ 143 , 87 , 65 , 255 ],[ 141 , 85 , 67 , 255 ],],
            [[ 154 , 98 , 62 , 255 ],[ 146 , 90 , 65 , 255 ],[ 141 , 85 , 67 , 255 ],[ 133 , 77 , 70 , 255 ],[ 128 , 72 , 72 , 255 ],[ 119 , 65 , 76 , 255 ],[ 114 , 60 , 77 , 255 ],[ 114 , 60 , 77 , 255 ],[ 112 , 57 , 79 , 255 ],[ 112 , 57 , 79 , 255 ],[ 109 , 55 , 79 , 255 ],[ 106 , 52 , 81 , 255 ],[ 104 , 50 , 81 , 255 ],[ 101 , 47 , 82 , 255 ],[ 99 , 45 , 84 , 255 ],],
            [[ 128 , 72 , 73 , 255 ],[ 117 , 62 , 76 , 255 ],[ 109 , 55 , 80 , 255 ],[ 99 , 45 , 84 , 255 ],[ 90 , 37 , 86 , 255 ],[ 79 , 27 , 91 , 255 ],[ 75 , 22 , 93 , 255 ],[ 72 , 20 , 94 , 255 ],[ 72 , 20 , 94 , 255 ],[ 70 , 17 , 95 , 255 ],[ 66 , 15 , 96 , 255 ],[ 66 , 15 , 96 , 255 ],[ 64 , 12 , 96 , 255 ],[ 61 , 10 , 98 , 255 ],[ 59 , 7 , 99 , 255 ],],
            [[ 101 , 47 , 83 , 255 ],[ 90 , 37 , 86 , 255 ],[ 79 , 27 , 91 , 255 ],[ 66 , 15 , 96 , 255 ],[ 56 , 5 , 100 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],],
            [[ 77 , 25 , 92 , 255 ],[ 64 , 12 , 97 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],],
            [[ 59 , 7 , 99 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],],
            [[ 53 , 2 , 101 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 51 , 0 , 102 , 255 ],[ 63 , 15 , 110 , 255 ],[ 100 , 60 , 135 , 255 ],[ 100 , 60 , 135 , 255 ],],
            [[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 199 , 182 , 202 , 255 ],[ 224 , 212 , 219 , 255 ],[ 250 , 243 , 236 , 255 ],[ 250 , 243 , 236 , 255 ],[ 250 , 243 , 236 , 255 ],[ 250 , 243 , 236 , 255 ],[ 250 , 243 , 236 , 255 ],],
        ])
        self.ingame=np.array([
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 235 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 165 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 9 , 35 , 254 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 219 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 188 , 255 ],[ 0 , 0 , 159 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 207 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 200 , 255 ],[ 0 , 0 , 162 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 251 , 255 ],[ 0 , 0 , 235 , 255 ],[ 0 , 0 , 213 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 197 , 255 ],[ 0 , 0 , 159 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 245 , 255 ],[ 0 , 0 , 223 , 255 ],[ 0 , 0 , 207 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 191 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 184 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 168 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 226 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 223 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 188 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 245 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 168 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 223 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 194 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 248 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 168 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 86 , 255 ],],
            [[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 213 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 194 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 95 , 255 ],[ 0 , 0 , 0 , 255 ],],
            [[ 0 , 0 , 207 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 239 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 165 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 124 , 255 ],[ 0 , 0 , 9 , 255 ],[ 0 , 0 , 0 , 255 ],],
            [[ 0 , 0 , 47 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 255 , 255 ],[ 0 , 0 , 207 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 204 , 255 ],[ 0 , 0 , 188 , 255 ],[ 0 , 0 , 153 , 255 ],[ 0 , 0 , 124 , 255 ],[ 0 , 0 , 9 , 255 ],[ 0 , 0 , 0 , 255 ],[ 0 , 0 , 0 , 255 ],],
        ])
        self.menu=np.array([
            [[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],],
            [[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],[ 220 , 159 , 32 , 255 ],],
            [[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],[ 220 , 159 , 33 , 255 ],],
            [[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],[ 220 , 159 , 34 , 255 ],],
            [[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],[ 220 , 160 , 35 , 255 ],],
            [[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],],
            [[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],[ 220 , 160 , 36 , 255 ],],
            [[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],[ 221 , 160 , 36 , 255 ],],
            [[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],[ 221 , 161 , 37 , 255 ],],
            [[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],],
            [[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],[ 221 , 161 , 38 , 255 ],],
            [[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],[ 221 , 161 , 39 , 255 ],],
            [[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],],
            [[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],[ 221 , 162 , 40 , 255 ],],
            [[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],[ 221 , 162 , 41 , 255 ],],
        ])
        self.win=np.array([
            [[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],],
            [[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],],
            [[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],],
            [[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],],
            [[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],[ 216 , 124 , 59 , 255 ],],
            [[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],],
            [[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],[ 216 , 125 , 60 , 255 ],],
            [[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],],
            [[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],[ 216 , 126 , 61 , 255 ],],
            [[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],],
            [[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],[ 217 , 127 , 62 , 255 ],],
            [[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],],
            [[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],[ 217 , 128 , 63 , 255 ],],
            [[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],[ 217 , 129 , 63 , 255 ],],
            [[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],[ 218 , 129 , 64 , 255 ],],
        ])
        #识别区域
        self.__reccognise_x=reccognise[0]
        self.__reccognise_y=reccognise[1]
        self.__reccognise_h=reccognise[2]
        self.__reccognise_w=reccognise[3]

    def __check_game_state_equal(self,arr1,arr2):
        pic1=Image.fromarray(np.uint8(arr1))
        pic2=Image.fromarray(np.uint8(arr2))
        print("------------")
        print('cos ',base_algorithm.cos_similar.image_similarity_vectors_via_numpy(pic1,pic2))
        print('hist ',hist_similar.pic_similar(pic1,pic2))
        print("three hist ",three_hist.classify_hist_with_split(pic1,pic2)[0])
        return False

    def get_game_state(self):
        game_state_pic=self.__env.capture_part(self.__reccognise_x,self.__reccognise_y,self.__reccognise_h,self.__reccognise_w)


        if self.__check_game_state_equal(game_state_pic,self.select):
            return SELECT
        if self.__check_game_state_equal(game_state_pic,self.fail):
            return FAIL
        if self.__check_game_state_equal(game_state_pic,self.ingame):
            return GAMEING
        if self.__check_game_state_equal(game_state_pic,self.menu):
            return MENU
        if self.__check_game_state_equal(game_state_pic,self.win):
            return WIN
        return PROCESSING


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
    game_env0=game_state(window_name,reccognise,resize_h,resize_w)

    print(game_env0.get_game_state())

    # while(1):
    #     print(game_env0.get_game_state())

