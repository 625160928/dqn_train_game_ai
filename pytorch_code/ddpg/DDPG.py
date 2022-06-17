"""
Created on  Feb 28 2021
@author: wangmeng
"""
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size ,init_w = 3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w,init_w)
        self.linear3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w = 3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # uniform_将tensor用从均匀分布中抽样得到的值填充。参数初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        #也用用normal_(0, 0.1) 来初始化的，高斯分布中抽样填充，这两种都是比较有效的初始化方式
        self.linear3.bias.data.uniform_(-init_w, init_w)
        #其意义在于我们尽可能保持 每个神经元的输入和输出的方差一致。
        #使用 RELU（without BN） 激活函数时，最好选用 He 初始化方法，将参数初始化为服从高斯分布或者均匀分布的较小随机数
        #使用 BN 时，减少了网络对参数初始值尺度的依赖，此时使用较小的标准差(eg：0.01)进行初始化即可

        #但是注意DRL中不建议使用BN

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0,0]

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 100000):#decay_period要根据迭代次数合理设置
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) *self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta* (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        #将经过tanh输出的值重新映射回环境的真实值内
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        #因为激活函数使用的是tanh，这里将环境输出的动作正则化到（-1，1）

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action

class DDPG(object):
    def __init__(self,action_dim, state_dim, hidden_dim):
        super(DDPG,self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 1e-2
        self.replay_buffer_size = 5000
        self.value_lr = 1e-3
        self.policy_lr = 1e-4


        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def save_model(self,log_dir):
        print('save models ')
        state = {'self.action_dim': self.action_dim,
                 'self.state_dim': self.state_dim,
                 'self.hidden_dim': self.hidden_dim,
                 'self.batch_size':self.batch_size,
                 'self.gamma':self.gamma,
                 'self.min_value': self.min_value,
                 'self.max_value':self.max_value,
                 'self.soft_tau':self.soft_tau,
                 'self.replay_buffer_size':self.replay_buffer_size,
                 'self.value_lr':self.value_lr,
                 'self.policy_lr':self.policy_lr,

                 'self.value_net': self.value_net.state_dict(),
                 'self.policy_net': self.policy_net.state_dict(),
                 'self.target_value_net': self.target_value_net.state_dict(),
                 'self.target_policy_net':self.target_policy_net.state_dict(),
                 'self.value_optimizer': self.value_optimizer.state_dict(),
                 'self.policy_optimizer':self.policy_optimizer.state_dict(),

                 'self.value_criterion': self.value_criterion.state_dict(),
                 'self.replay_buffer':self.replay_buffer
                 }
        torch.save(state, log_dir)

    def load_model(self,log_dir):
        print('load  models ')
        checkpoint = torch.load(log_dir)
        self.action_dim=checkpoint['self.action_dim']
        self.state_dim=checkpoint['self.state_dim']
        self.hidden_dim=checkpoint['self.hidden_dim']
        self.batch_size=checkpoint['self.batch_size']
        self.gamma=checkpoint['self.gamma']
        self.min_value=checkpoint['self.min_value']
        self.max_value=checkpoint['self.max_value']
        self.soft_tau=checkpoint['self.soft_tau']
        self.replay_buffer_size=checkpoint['self.replay_buffer_size']
        self.value_lr=checkpoint['self.value_lr']
        self.policy_lr=checkpoint['self.policy_lr']


        self.value_net.load_state_dict(checkpoint['self.value_net'])
        self.policy_net.load_state_dict(checkpoint['self.policy_net'])
        self.target_value_net.load_state_dict(checkpoint['self.target_value_net'])
        self.target_policy_net.load_state_dict(checkpoint['self.target_policy_net'])
        self.value_optimizer.load_state_dict(checkpoint['self.value_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['self.policy_optimizer'])
        self.value_criterion.load_state_dict(checkpoint['self.value_criterion'])


        self.replay_buffer=checkpoint['self.replay_buffer']




    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

    def push_memory(self,state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) > self.batch_size:
            self.ddpg_update()


def plot(frame_idx, rewards):
    # plt.figure(figsize=(20,5))
    # plt.subplot(131)
    # plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.pause(0.01)


def main(show=True):
    env = gym.make("Pendulum-v0")
    env = NormalizedActions(env)

    ou_noise = OUNoise(env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    ddpg = DDPG(action_dim, state_dim, hidden_dim)

    max_frames = 12000
    max_steps = 1000
    frame_idx = 0
    rewards = []
    save_path='./models/ddpg_model_'


    while frame_idx < max_frames:
        frame_idx += 1
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            if show and frame_idx>250:
                env.render()
            action = ddpg.policy_net.get_action(state)
            # print('1 ',action,type(action))
            # action = ou_noise.get_action(action, step)
            action=np.array([action])
            # print('2 ' ,action,type(action))

            next_state, reward, done, _ = env.step(action)

            ddpg.push_memory(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward


            if done:
                # print("break ")
                break

        if frame_idx%50==0:
            ddpg.save_model(save_path+str(frame_idx)+'.pth')
        rewards.append(episode_reward)
        print(frame_idx,episode_reward)
        if frame_idx % 30 == 0:
            plot(frame_idx, rewards)
    env.close()

if __name__ == '__main__':
    main(show=True)
    # main(show=False)