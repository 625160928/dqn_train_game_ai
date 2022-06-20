import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import  Counter
from collections import deque
import matplotlib.pyplot as plt


from dqn.dqn4 import DQN

USE_CUDA = torch.cuda.is_available()
# USE_CUDA=False

# USE_CUDA=False
#将变量放到cuda上
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()\
    if USE_CUDA else autograd.Variable(*args, **kwargs)


def run_dqn():
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    #训练基础参数
    observation_space = env.observation_space.shape[0]
    action_sapce = env.action_space.n
    capacity = 1000

    #训练次数
    num_frames = 3000

    #数据记录
    all_rewards = []
    x_axis1 = []

    # import os
    # print(os.path.abspath('.'))

    #加载模型
    model = DQN(observation_space, action_sapce, capacity)
    model.load_model('./dqn/models/' + 'dqn4_model_nr_2400.pth')
    if USE_CUDA:
        model = model.cuda()

    #开始训练、测试
    for frame_idx in range( num_frames):
        state = env.reset()
        episode_reward = 0
        count = 0
        while True:
            count += 1

            epsilon=0

            action = model.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)

            #重计算奖励
            w1 = 1
            w2 = 0
            w3 = 1
            w4 = 0
            w = w1 + w2 + w3 + w4
            r1 = np.clip(2.4 - abs(next_state[0]), 0, 2.4) / 2.4
            r2 = np.clip(0.5 - abs(next_state[1]), 0, 0.5) / 0.5
            r3 = np.clip(0.418 - abs(next_state[2]), 0, 0.418) / 0.418
            r4 = np.clip(0.5 - abs(next_state[3]), 0, 0.5) / 0.5
            new_reward = (r1 * w1 + r2 * w2 + r3 * w3 + r4 * w4) / w

            model.push_memory(state, action, new_reward, next_state, done)

            # print(next_state,new_reward)

            state = next_state
            episode_reward += new_reward

            env.render()

            if done:
                break

        #记录数据
        x_axis1.append(frame_idx)
        all_rewards.append(episode_reward)


        print('id= ', frame_idx, ' live_time= ', count, ' reward= ', episode_reward)


def main():
    run_dqn()

if __name__ == '__main__':
    main()