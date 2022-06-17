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
import os
import pandas as pd
import csv

USE_CUDA=False
#将变量放到cuda上
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DQN(nn.Module):
    def __init__(self, observation_space, action_space, capacity):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_space,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        self.capacity=capacity
        self.observation_space = observation_space
        self.action_sapce = action_space
        self.losses = []

        self.optimizer = optim.Adam(self.parameters())

        self.batch_size = 32
        self.gamma = 0.99

        #deque模块是python标准库collections中的一项，它提供了两端都可以操作的序列，其实就是双向队列，
        #可以从左右两端增加元素，或者是删除元素。如果设置了最大长度，非输入端的数据会逐步移出窗口。
        self.buffer = deque (maxlen = self.capacity)

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            #如果使用的是GPU，这里需要把数据丢到GPU上
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)#volatile的作用是作为指令关键字，确保本条指令不会因编译器的优化而省略，且要求每次直接读值。
            #.squeeze() 把数据条目中维度为1 的删除掉
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
            #max(1)返回每一行中最大值的那个元素，且返回其索引,max(0)是列
            #max()[1]只返回最大值的每个索引，max()[0]， 只返回最大值的每个数

            action = action.cpu().numpy()#从网络中得到的tensor形式，因为之后要输入给gym环境中，这里把它放回cpu，转为数组形式
            action =int(action)
        else:
            action = random.randrange(self.action_sapce)#返回指定递增基数集合中的一个随机数，基数默认值为1。
        return action

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.sample_memory()
        #通通丢到GPU上去
        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self(state)
        next_q_values = self(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        #gather可以看作是对q_values的查询，即元素都是q_values中的元素，查询索引都存在action中。输出大小与action.unsqueeze(1)一致。
        #dim=1,它存放的都是第1维度的索引；dim=0，它存放的都是第0维度的索引；
        #这里增加维度主要是为了方便gather操作，之后再删除该维度
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(np.array(loss.data.cpu()))

        return loss

    def push_memory (self, state, aciton, reward, next_state, done):
        state = np.expand_dims(state,0)
        #这里增加维度的操作是为了便于之后使用concatenate进行拼接
        next_state = np.expand_dims(next_state,0)
        self.buffer.append((state, aciton, reward, next_state, done))
        if len(self.buffer)>=self.batch_size:
            self.compute_td_loss()

    def sample_memory(self):
        # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        state , action, reward, next_state, done = zip(*random.sample(self.buffer, self.batch_size))
        #最后使用concatenate对数组进行拼接，相当于少了一个维度
        return np.concatenate(state),  action, reward, np.concatenate(next_state), done

    def save_dqn_model(self,log_dir):
        # print('save models ')
        state = {'layers': self.layers.state_dict(),
                 'capacity': self.capacity,
                 'observation_space':self.observation_space,
                 'action_sapce':self.action_sapce,
                 'losses':self.losses,
                 'optimizer': self.optimizer.state_dict(),
                 'batch_size':self.batch_size,
                 'gamma':self.gamma,
                 'buffer':self.buffer,
                 }
        torch.save(state, log_dir)

    def load_dqn_model(self,log_dir):
        # print('load  models ')
        checkpoint = torch.load(log_dir)
        self.layers.load_state_dict(checkpoint['layers'])
        self.optimizer.load_state_dict(checkpoint[ 'optimizer'])
        self.capacity=checkpoint['capacity']
        self.observation_space=checkpoint['observation_space']
        self.action_sapce=checkpoint[ 'action_sapce']
        self.losses=checkpoint['losses']
        self.batch_size=checkpoint['batch_size']
        self.gamma=checkpoint['gamma']
        self.buffer=checkpoint[ 'buffer']



def run_model(model,times,env,draw=False):
    total_score=0
    for frame_idx in range(times):
        state = env.reset()
        episode_reward = 0
        count=0
        while True:
            count+=1
            epsilon = 0

            action = model.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)

            w1=1
            w2=1
            w3=2
            w4=1
            w=w1+w2+w3+w4


            r1=np.clip(2.4-abs(next_state[0]),0,2.4)/2.4
            r2=np.clip(0.5-abs(next_state[1]),0,0.5)/0.5
            r3=np.clip(0.418-abs(next_state[2]),0,0.418)/0.418
            r4=np.clip(0.5-abs(next_state[3]),0,0.5)/0.5
            new_reward=(r1*w1+r2*w2+r3*w3+r4*w4)/w
            model.push_memory(state, action, new_reward, next_state, done)

            # print(next_state,new_reward)


            state = next_state
            episode_reward += new_reward
            # print(frame_idx)

            if draw:
                env.render()

            if done:
                break
        total_score+=count
        # print(count,episode_reward)
    return total_score

'''
['dqn4_model_nr_2500.pth', 40065]
['dqn4_model_nr1131_2000.pth', 40901]
['dqn4_model_nr_4000.pth', 43761]
['dqn4_model_nr_1200.pth', 43863]
['dqn4_model_nr1131_1900.pth', 44654]
['dqn4_model_nr_2400.pth', 46469]

['dqn4_model_nr1131_1900.pth', 50000]
['dqn4_model_nr_2400.pth', 50000]

'''
def save_file(dict_map,col,file_path):
    pd.DataFrame(data=dict_map, columns=col).to_csv(file_path , index=False)

def load_file(col,file_path):
    dim=dict()
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if  line==col:
                continue
            dim[line[0]]=float(line[1])
    # print(dim)
    return dim

def try_all():
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    observation_space = env.observation_space.shape[0]
    action_sapce = env.action_space.n
    capacity=1000
    test_times=100

    src_path="./models/"
    save_path=src_path+'anylize_map.csv'
    col=['name','live_time']
    files_path=os.listdir(src_path)

    model_jdu=[]

    dict_map=load_file(col,save_path)



    for model_path in files_path:
        k=model_path.split('.')
        if k[1]!='pth':
            continue


        if model_path in dict_map:
            score=dict_map[model_path]
        else:
            model = DQN (observation_space, action_sapce,capacity)
            if USE_CUDA:
                model = model.cuda()
            model.load_dqn_model(src_path+model_path)
            score=run_model(model,test_times,env)
            dict_map[model_path]=score
            save_file(model_jdu,col,save_path)

        model_jdu.append([model_path,score])



        print(model_path,score,score/test_times)

        # load_file(col,save_path)

    print("=====================")
    model_jdu.sort(key=lambda x:x[1])
    for m in model_jdu:
        print(m)

def try_once(model_path):
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    observation_space = env.observation_space.shape[0]
    action_sapce = env.action_space.n
    capacity = 1000
    test_times = 100


    model = DQN(observation_space, action_sapce, capacity)
    if USE_CUDA:
        model = model.cuda()

    model.load_dqn_model(model_path)

    score = run_model(model, test_times, env,draw=True)


def main():
    # try_all()
    try_once('./models/'+'dqn4_model_nr1131_1900.pth')


if __name__ == '__main__':
    main()
