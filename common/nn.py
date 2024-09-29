import numpy as np
import torch
import torch.nn.functional as f
from torch import nn


class CNN(nn.Module):
    """默认期望输入形状 (n, 3, 308, 308)，其中 n 为批大小"""

    def __init__(self, n_hidden: int, action: int):
        super(CNN, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 定义第三个卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 定义一个全连接层, 若需要自定义的输入形状，则修改in_features的值
        self.fc = nn.Linear(in_features=n_hidden, out_features=512)
        # 定义输出层
        self.out = nn.Linear(in_features=512, out_features=action)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.fc(x.view(x.size(0), -1)))
        return self.out(x)


class DQN:
    def __init__(self,
                 n_hidden: int,
                 n_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon: float,
                 target_update_frequency: int,
                 device: torch.device):
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update_frequency
        self.device = device
        self.count = 0
        self.q_net = CNN(self.n_hidden, self.n_actions)
        self.target_q_net = CNN(self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def choose_action(self, state: torch.Tensor, train_mode=True):
        """
        根据当前状态选择行为
        :param state: 模型输入
        :param train_mode: 训练模式
        :return: 行为枚举值
        """
        if train_mode:
            # 只有训练模式以 1 - self.epsilon 的概率随机选择某一行为
            if np.random.random() > self.epsilon:
                action = np.random.randint(self.n_actions)
                return action
        actions_value = self.q_net(state)
        action = actions_value.argmax().item()
        return action

    def train(self, transition_dict: dict):
        """
        DQN网络训练过程
        :param transition_dict: 经验缓冲池抽取的随机采样
        """
        # 从字典中提取并处理状态、动作、奖励、下一个状态和完成标志
        states = transition_dict['states'].clone().detach().to(dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = transition_dict['next_states'].clone().detach().to(dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)
        # 计算当前状态下，根据当前策略选择特定动作的Q值
        q_values = self.q_net(states).gather(1, actions)
        # 使用目标网络计算下一个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 计算Q学习的目标值，如果是最终状态，则只考虑立即奖励
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算DQN的损失，这里使用均方误差损失函数
        dqn_loss = torch.mean(f.mse_loss(q_values, q_targets))
        # 优化器重置梯度
        self.optimizer.zero_grad()
        # 反向传播损失，更新网络权重
        dqn_loss.backward()
        self.optimizer.step()
        # 每隔一定的步数更新目标网络的权重
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        # 增加训练步数的计数器
        self.count += 1
