import gym
import math
import random
import datetime
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from copy import deepcopy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# File name
fmt = '%Y%m%d%H%M%S'
stamp = datetime.datetime.now().strftime(fmt)
file_name = 'episode_durations' + stamp + '.p'

# Disable cuda for now
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append(Transition(state, action, reward, next_state))

    def get_minibatch(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 48)
        self.fc_output = nn.Linear(48, 2)

    def forward(self, x):
        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.fc_output(hidden2)
        return output

class DQN():
    def __init__(self, env, batch_size, gamma, eps_max, eps_min,
                 eps_decay, model, replay_memory, optimizer, max_episodes=200):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.model = model
        self.replay_memory = replay_memory
        self.optimizer = optimizer
        self.episode_durations = []
        self.last_sync = 0
        self.max_episodes = max_episodes

    def get_epsilon(self, t):
        return max(self.eps_min, min(self.eps_max, 1.0 - math.log10((t + 1) * self.eps_decay)))

    def select_action(self, state, t):
        if random.random() > self.get_epsilon(t):
            return self.model(Variable(state, volatile=True).type(FloatTensor)).data.max(0)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(2)]])

    def optimize_model(self):
        transitions = self.replay_memory.get_minibatch(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        non_final_states = torch.stack([s for s in batch.next_state if s is not None])
        non_final_next_states_tensor = Variable(non_final_states, requires_grad=False)

        state_batch = Variable(torch.stack(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute max_a Q(s_{t+1}, a)
        next_state_values = Variable(torch.zeros(len(batch.state)).type(Tensor), requires_grad=False)
        next_state_values[non_final_mask] = self.model(non_final_next_states_tensor).max(1)[0]
        # y, if done only reward
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        scores = deque(maxlen=100)
        while len(scores) is 0 or np.mean(scores) < 195:
            state = torch.from_numpy(np.ascontiguousarray(self.env.reset(), dtype=np.float32))
            for t in count():
                action = self.select_action(state, len(self.episode_durations))
                next_state, reward, done, _ = self.env.step(action[0, 0])
                next_state = torch.from_numpy(np.ascontiguousarray(next_state, dtype=np.float32))

                if done:
                    next_state = None

                self.replay_memory.push(state, action, next_state, Tensor([reward]))
                state = next_state
                self.optimize_model()

                if done:
                    duration = t + 1
                    self.episode_durations.append(duration)
                    scores.append(duration)
                    # pickle.dump(self.episode_durations, open(file_name, 'wb'))
                    if np.mean(scores) > 40:
                        print(np.mean(scores))
                    break
        self.env.close()



if __name__ == '__main__':
    dqn_model = Net()
    dqn_optimizer = optim.Adam(dqn_model.parameters(), lr=0.01, weight_decay=0.01)

    dqn = DQN(env=gym.make('CartPole-v0'), batch_size=64, gamma=0.99, eps_max=1, eps_min=0.01,
              eps_decay=0.999, model=dqn_model, replay_memory=ReplayMemory(10000), optimizer=dqn_optimizer)

    dqn.run()
    print(len(dqn.episode_durations))

