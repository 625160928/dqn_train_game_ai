import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# 超参数
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9  # 随机选取的概率，如果概率小于这个随机数，就采取greedy的行为
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
# 导入openAI gym实验的模拟场所，'CartPole-v0'表示倒立摆的实验
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 小车的动作
N_STATES = env.observation_space.shape[0]  # 实验环境的状态
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        # DQN是Q-Leaarning的一种方法，但是有两个神经网络，一个是eval_net一个是target_net
        # 两个神经网络相同，参数不同，是不是把eval_net的参数转化成target_net的参数，产生延迟的效果
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库
        # 记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # 优化器和损失函数

    # 接收环境中的观测值，并采取动作
    def choose_action(self, x):
        # x为观测值
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            # 随机值得到的数有百分之九十的可能性<0.9,所以该if成立的几率是90%
            # 90%的情况下采取actions_value高的作为最终动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:
            # 其他10%采取随机选取动作
            action = np.random.randint(0, N_ACTIONS)  # 从动作中选一个动作
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

        # 记忆库，存储之前的记忆，学习之前的记忆库里的东西

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新,每隔TARGET_REPLACE_ITE更新一下
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # targetnet是时不时更新一下，evalnet是每一步都更新

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # 打包记忆，分开保存进b_s，b_a，b_r，b_s
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0]  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer.step()

    def save_dqn_model(self,log_dir):
        print('save model ',self.learn_step_counter)
        state = {'eval_net_model': self.eval_net.state_dict(),'target_net_model': self.target_net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'loss_func': self.loss_func,'memory':self.memory,'memory_counter':self.memory_counter,
                 'learn_step_counter':self.learn_step_counter}
        torch.save(state, log_dir)

    def load_dqn_model(self,log_dir):
        print('load  model ')
        checkpoint = torch.load(log_dir)
        self.eval_net.load_state_dict(checkpoint['eval_net_model'])
        self.target_net.load_state_dict(checkpoint['target_net_model'])
        self.optimizer.load_state_dict(checkpoint[ 'optimizer'])
        self.loss_func=checkpoint['loss_func']
        self.memory_counter=checkpoint['memory_counter']
        self.learn_step_counter=checkpoint[ 'learn_step_counter']
        self.memory=checkpoint['memory']

if __name__ == '__main__':
    dqn = DQN()
    print('\nCollection experience...')
    model_path='./dqn2_mode.pth'
    save_path='./dqn3_models/dqn3_mode_'
    dqn.load_dqn_model(model_path)
    train_time=[]
    train_eff=[]
    for i_episode in range(40000):
        s = env.reset()  # 得到环境的反馈，现在的状态
        ep_r = 0
        while True:
            env.render()  # 环境渲染，可以看到屏幕上的环境
            a = dqn.choose_action(s)  # 根据dqn来接受现在的状态，得到一个行为
            s_, r, done, info = env.step(a)  # 根据环境的行为，给出一个反馈

            # 修改 reward, 使 DQN 快速学习
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态

            ep_r += r

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break

            s = s_  # 现在的状态赋值到下一个状态上去


        train_time.append(i_episode)
        train_eff.append(ep_r)

        if i_episode % 10 == 0:
            plt.clf()
            plt.plot(train_time,train_eff)
            plt.pause(0.01)

        if i_episode%50==0:
            dqn.save_dqn_model(save_path+str(i_episode)+'.pth')