#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
try:
    from . import wrapped_flappy_bird as game
except Exception:
    import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
'''
先观察一段时间（OBSERVE = 1000 不能过大），
获取state(连续的4帧) => 进入训练阶段（无上限）=> action

'''
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions 往上  往下
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon 探索
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

# GAME = 'bird' # the name of the game being played for log files
# ACTIONS = 2 # number of valid actions
# GAMMA = 0.99 # decay rate of past observations
# OBSERVE = 100000. # timesteps to observe before training
# EXPLORE = 2000000. # frames over which to anneal epsilon
# FINAL_EPSILON = 0.0001 # final value of epsilon
# INITIAL_EPSILON = 0.0001 # starting value of epsilon
# REPLAY_MEMORY = 50000 # number of previous transitions to remember
# BATCH = 32 # size of minibatch
# FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# padding = ‘SAME’=> new_height = new_width = W / S （结果向上取整）
# padding = ‘VALID’=> new_height = new_width = (W – F + 1) / S （结果向上取整）
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
"""
 数据流：80 * 80 * 4  
 conv1(8 * 8 * 4 * 32, Stride = 4) + pool(Stride = 2)-> 10 * 10 * 32(height = width = 80/4 = 20/2 = 10)
 conv2(4 * 4 * 32 * 64, Stride = 2) -> 5 * 5 * 64 + pool(Stride = 2)-> 3 * 3 * 64
 conv3(3 * 3 * 64 * 64, Stride = 1) -> 3 * 3 * 64 = 576
 576 在定义h_conv3_flat变量大小时需要用到，以便进行FC全连接操作
"""

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([576, 512])
    b_fc1 = bias_variable([512])
    # W_fc1 = weight_variable([1600, 512])
    # b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 576])
    #h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    #h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    # reduction_indices = axis  0 : 列  1: 行
    # 因 y 是数值，而readout: 网络模型预测某个行为的回报 大小[1, 2] 需要将readout 转为数值，
    # 所以有tf.reduce_mean(tf.multiply(readout, a), axis=1) 数组乘法运算，再求均值。
    # 其实，这里readout_action = tf.reduce_mean(readout, axis=1) 直接求均值也是可以的。
    readout_action = tf.reduce_mean(tf.multiply(readout, a), axis=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # 创建队列保存参数
    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    #cv2.imwrite('x_t.jpg',x_t)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    tf.summary.FileWriter("tensorboard/", sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    """
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    """
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        # 预测结果（当前状态不同行为action的回报，其实也就 往上，往下 两种行为）
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            # 加入一些探索，比如探索一些相同回报下其他行为，可以提高模型的泛化能力。
            # 且epsilon是随着模型稳定趋势衰减的，也就是模型越稳定，探索次数越少。
            if random.random() <= epsilon:
                # 在ACTIONS范围内随机选取一个作为当前状态的即时行为
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                # 输出 奖励最大就是下一步的方向
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon 模型稳定，减少探索次数。
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        # 先将尺寸设置成 80 * 80，然后转换为灰度图
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        # x_t1 新得到图像，二值化 阈值：1
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        # 取之前状态的前3帧图片 + 当前得到的1帧图片
        # 每次输入都是4幅图像
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        # s_t: 当前状态（80 * 80 * 4）
        # a_t: 即将行为 （1 * 2）
        # r_t: 即时奖励
        # s_t1: 下一状态
        # terminal: 当前行动的结果（是否碰到障碍物 True => 是 False =>否）
        # 保存参数，队列方式，超出上限，抛出最左端的元素。
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # 获取batch = 32个保存的参数集
            minibatch = random.sample(D, BATCH)
            # get the batch variables
            # 获取j时刻batch(32)个状态state
            s_j_batch = [d[0] for d in minibatch]
            # 获取batch(32)个行动action
            a_batch = [d[1] for d in minibatch]
            # 获取保存的batch(32)个奖励reward
            r_batch = [d[2] for d in minibatch]
            # 获取保存的j + 1时刻的batch(32)个状态state
            s_j1_batch = [d[3] for d in minibatch]
            # readout_j1_batch =>(32, 2)
            y_batch = []
            readout_j1_batch = sess.run(readout, feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:  # 碰到障碍物，终止
                    y_batch.append(r_batch[i])
                else: # 即时奖励 + 下一阶段回报
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
            # 根据cost -> 梯度 -> 反向传播 -> 更新参数
            # perform gradient step
            # 必须要3个参数，y, a, s 只是占位符，没有初始化
            # 在 train_step过程中，需要这3个参数作为变量传入
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1  # state 更新
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("terminal", terminal, \
              "TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
