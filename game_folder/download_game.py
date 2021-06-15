# -*- coding: utf-8 -*-

import re
import os
import requests

# 基础url
host_url = 'http://www.4399.com'
swfbase_url = 'http://sda.4399.com/4399swf'

#https://blog.csdn.net/alobjgo278266549/article/details/101309441
# # 需要的正则表达式
# tmp_pat = re.compile(r'<ul class="tm_list">(.*?)</ul>',re.S)
# game_pat = re.compile( r'<li><a href="(/flash.*?)"><img alt=.*?src=".*?"><b>(.*?)</b>.*?</li>', re.S )
# swf_pat = re.compile(r'_strGamePath="(.*?swf)"',re.S)
#
# def main():
#
#
#     if not os.path.exists('./swf'):
#         os.mkdir(r'./swf')
#
#
#     game_html = requests.get(hw_url)
#     game_html.encoding = 'gb2312'
#
#     # print(game_html.text)
#
#     tt = tmp_pat.search(game_html.text,re.S).group(1)
#     # print(tt)
#     game_list = game_pat.findall(tt)
#     # print(game_list)
#     for l in game_list:
#         # print l[0], l[1]
#         if l[1]=='老爹饼干圣代店':
#             print("----------------------------")
#             print(host_url + l[0])
#             game_page = requests.get(host_url + l[0]).text
#             print( game_page)
#
#             src_url = swf_pat.search(game_page)
#             if src_url == None:
#                 continue
#             print(l[1],' ',src_url.group(1))
#
#             src0=swfbase_url + src_url.group(1)
#             print("当前网址是 ",src0)
#             src = requests.get( src0 ).content
#             print ("正在保存游戏:" , l[1] )
#             open( "./swf/"+ l[1] + ".swf", "wb" ).write( src )
#             break

#根据网页网址，获取游戏
def download_game(url):
    #获取网页源代码
    game_page = requests.get(url).text
    # game_page.encoding = 'gb2312'
    if not os.path.exists('./game_src'):
        os.mkdir(r'./game_src')
    #找到网页中含有游戏标题的字符
    start =game_page.find("game_title=")
    #标记标题起始位置
    p1=start+14
    #通过向后遍历寻找双引号获取标题结束的位置
    p2=p1+1
    while(game_page[p2]!="\""):
        p2+=1
    #保存标题
    game_title= game_page[p1:p2]

    #和获取标题一样的方法获取游戏本体的网址
    start =game_page.find("_strGamePath=")
    p1=start+14
    p2=p1+1
    while(game_page[p2]!="\""):
        p2+=1
    #4399游戏本体的网址是由两部分组成，基址+偏移量
    #我们之前找的都是偏移量，接上基址就是正确的网址
    src_url=swfbase_url + game_page[p1:p2]

    file_type='.'+game_page[p1:p2].split('/')[-1].split('.')[-1]
    # print(file_type)
    if file_type=='.swf':
        #获取本体
        src = requests.get( src_url).content
        print(game_title)
        #保存
        open( "./game_src/"+ game_title+ file_type, "wb" ).write( src )
    else:
        print("只有swf文件才能下载")


if __name__ == '__main__':

    #森林冰火人2选关版  http://www.4399.com/flash/175700.htm#search3-9af1

    url='http://www.4399.com/flash/175700.htm#search3-9af1'

    download_game(url)

