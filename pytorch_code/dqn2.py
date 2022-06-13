import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/reinforcement_q_learning.md

env = gym.make('CartPole-v0').unwrapped

# 建立 matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """保存变换"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 线性输入连接的数量取决于conv2d层的输出，因此需要计算输入图像的大小。
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2) # 448 或者 512

    # 使用一个元素调用以确定下一个操作，或在优化期间调用批处理。返回张量
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # 车子的中心

def get_screen():
    # 返回 gym 需要的400x600x3 图片, 但有时会更大，如800x1200x3. 将其转换为torch (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 车子在下半部分，因此请剥去屏幕的顶部和底部。
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 去掉边缘，这样我们就可以得到一个以车为中心的正方形图像。
    screen = screen[:, :, slice_range]
    # 转化为 float, 重新裁剪, 转化为 torch 张量(这并不需要拷贝)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 重新裁剪,加入批维度 (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1）将为每行的列返回最大值。max result的第二列是找到max元素的索引，因此我们选择预期回报较大的操作。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 平均 100 次迭代画一次
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 暂定一会等待屏幕更新
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置批样本(有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。这会将转换的批处理数组转换为批处理数组的转换。
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。这些是根据策略网络对每个批处理状态所采取的操作。
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算下一个状态的V(s_{t+1})。非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；选择max(1)[0]的最佳奖励。这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # print("next_state_values[non_final_mask] ",target_net(non_final_next_states).max(1)[0].detach())
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算期望 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算 Huber 损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == '__main__':
    plt.ion()

    # 如果使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ",device)

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    #获取屏幕大小，以便我们可以根据从ai-gym返回的形状正确初始化层。这一点上的典型尺寸接近3x40x90，这是在get_screen(）中抑制和缩小的渲染缓冲区的结果。
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    policy_net = DQN(screen_height, screen_width).to(device)
    target_net = DQN(screen_height, screen_width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0


    episode_durations = []



    num_episodes = 50
    for i_episode in range(num_episodes):
        print('i_episode ',i_episode)
        # 初始化环境和状态
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # 选择并执行动作
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # 观察新状态
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # 在内存中储存当前参数
            memory.push(state, action, next_state, reward)

            # 进入下一状态
            state = next_state

            # 记性一步优化 (在目标网络)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        #更新目标网络, 复制在 DQN 中的所有权重偏差
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
