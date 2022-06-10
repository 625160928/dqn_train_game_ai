
import torch
import torch.nn as nn
import numpy as np
from Environment import Maze

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)   # input
        self.fc1.weight.data.normal_(0, 0.01)   # 运用二次分布随机初始化，以得到更好的值
        self.out = nn.Linear(50, n_actions)   # output
        self.out.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN():
    def __init__(self,
                 n_states,
                 n_actions,
                 batch_size=32,
                 learning_rate=0.01,
                 epsilon=0.9,
                 gamma=0.9,
                 target_replace_iter=100,
                 memory_size=2000):

        # 生成两个结构相同的神经网络eval_net和target_net
        self.eval_net, self.target_net = Net(n_states, n_actions),\
                                         Net(n_states, n_actions)

        self.n_states = n_states  # 状态维度
        self.n_actions = n_actions  # 可选动作数
        self.batch_size = batch_size  # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # 贪婪系数
        self.gamma = gamma  # 回报衰减率
        self.memory_size = memory_size  # 记忆库的规格
        self.taget_replace_iter = target_replace_iter  # target网络延迟更新的间隔步数
        self.learn_step_counter = 0  # 在计算隔n步跟新的的时候用到，说明学习到多少步了
        self.memory_counter = 0  # 用来计算存储索引
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2))  # 初始化记忆库，存的量是两个state在加上一个reward和action
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)  # 网络优化器
        self.loss_func = nn.MSELoss()  # 网络的损失函数

    def choose_action(self, x):  # x是观测值
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:  # greedy概率有eval网络生成动作,此时选择actions_value最大的动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1]
        else:  # （1-greedy）概率随机选择动作
            action = np.random.randint(0, self.n_actions)
        return action

    # 将信息存储到经验池
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))   # 将信息捆在一起
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size  # 如果memory_counter超过了memory_size的上限，就重新开始索引
        self.memory[index, :] = transition  # 将信息存到相对应的位置
        self.memory_counter += 1

    def learn(self):
        # 判断target net什么时候更新
        if self.learn_step_counter % self.taget_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将eval_net中的state更新到target_net中

        # 采用mini-batch去更新(从记忆库中随机抽取一些记忆)
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        # 获得q_eval、q_target，计算loss
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        # 将loss反向传递回去，更新eval网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env = Maze()
    dqn = DQN(env.n_states, env.n_actions)

    print('Collecting experience...')
    for i_episode in range(400):
        s = env.reset()                 # 重置初始状态
        ep_r = 0
        while True:
            env.render()                # 刷新画面
            a = dqn.choose_action(s)    # 选择动作
            s_, r, done = env.step(a)   # 执行动作，获得下一个状态s_，回报r，是否结束标记done
            dqn.store_transition(s, a, r, s_)   # 存储 一步 的信息
            ep_r += r                   # ep_r，一轮中的总回报
            if dqn.memory_counter > dqn.memory_size:    # 当记忆库存满（非必要等到存满）的时候，开始训练
                dqn.learn()
                if done:
                    if i_episode%20==0:
                        print('Ep: ', i_episode + 1, '| Ep_r: ', round(ep_r, 2))
            if done:                    # 如果done（智能到达终点/掉入陷阱），结束本轮
                break
            s = s_

    # 测试部分
    print('Testing . . .')
    # dqn.epsilon = 1
    rs = []
    for state in range(50): # 打算循环测试50次测一测平均回报
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = dqn.choose_action(s)
            s_, r, done = env.step(a)
            ep_r += r
            # 测试阶段就不再有存储和学习了
            if done:
                print(ep_r)
                rs.append(ep_r)
                break
            s = s_

    env.close()
	# 测试50次之后，输出一下平均每轮的回报
    print(np.average(rs))