import gym
import math
import random
import datetime
import numpy as np
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
from textwrap import wrap
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
fmt = '%Y-%m-%d_%H:%M:%S'
stamp = datetime.datetime.now().strftime(fmt)
file_name = 'episode_durations' + stamp + '.p'
img_name = 'figure_' + stamp + '.png'
avg_action = 'avg_action' + stamp + '.p'

# Disable cuda for now
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="skip k number of frames", type=int, default=0)
    parser.add_argument("--save", help="save episode durations", action='store_true')
    return parser.parse_args()

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

class DqnNet(nn.Module):
    def __init__(self):
        super(DqnNet, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc_output = nn.Linear(24, 2)

    def forward(self, x):
        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.fc_output(hidden2)
        return output

class EpsilonGreedy():
    def __init__(self, eps_max, eps_min, decay):
        self.max = eps_max
        self.min = eps_min
        self.decay = decay

    def get_epsilon(self, steps):
        return self.min + (self.max - self.min) * math.exp(-1. * steps / self.decay)

    def __str__(self):
        return 'EPSILON GREEDY: min:' + str(self.min) + ' max:' + str(self.max) + ' decay:' + str(self.decay)

class LinearEpsilonGreedy():
    def __init__(self, epx_max, eps_min, decay):
        self.max = epx_max
        self.min = eps_min
        self.decay = decay

    def get_epsilon(self, steps):
        return max(self.min, self.max / max(1, (steps * self.decay)))

    def __str__(self):
        return 'EPSILON GREEDY: min:' + str(self.min) + ' max:' + str(self.max) + ' decay:' + str(self.decay)

class DQN():
    def __init__(self, env, batch_size, gamma, eps_greedy, model, replay_memory, optimizer,
                 max_episodes=200, save_episodes=False, skip_k=0, max_iter=1000):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_greedy = eps_greedy
        self.model = model
        self.replay_memory = replay_memory
        self.optimizer = optimizer
        self.episode_durations = []
        self.last_sync = 0
        self.max_episodes = max_episodes
        self.steps_done = 0
        self.save_episodes = save_episodes
        self.skip_k = skip_k
        # For stopping condition
        self.action_queue = deque(maxlen=1000)
        self.action_values = []
        self.max_iter = max_iter

    def __str__(self):
        return 'DQN: batch_size:' + str(self.batch_size) + ' gamma:' + str(self.gamma) + ' k:' + str(self.skip_k)

    def select_action(self, state, t):
        if random.random() > self.eps_greedy.get_epsilon(t):
            actions = self.model(Variable(state, volatile=True).type(FloatTensor)).data
            self.action_queue.append(actions.max())
            return actions.max(0)[1].view(1, 1)
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
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def handle_episode_results(self, duration):
        avg_action_value = np.mean(self.action_queue)
        self.episode_durations.append(duration)
        self.action_values.append(avg_action_value)
        if self.save_episodes:
            pickle.dump(self.episode_durations, open(file_name, 'wb'))
            pickle.dump(self.action_values, open(avg_action, 'wb'))

        # if len(self.episode_durations) % 20 == 0:
        #     print(avg_action_value)


    def run(self):
        scores = deque(maxlen=100)
        for i in range(self.max_iter):
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

                self.steps_done += 1

                if done:
                    score = t+1
                    self.handle_episode_results(score)
                    scores.append(score)
                    break

            # if np.mean(scores) >= 195 and i >= 100:
            #     return i


        self.env.close()



if __name__ == '__main__':
    args = parse_cl_args()

    dqn_model = DqnNet()
    lr=0.0003
    dqn_optimizer = optim.Adam(dqn_model.parameters(), lr=lr)
    dqn_eps = EpsilonGreedy(eps_max=1, eps_min=0.01, decay=200)
    dqn_linear_eps = LinearEpsilonGreedy(epx_max=1, eps_min=0.01, decay=0.01)
    eps = dqn_linear_eps

    max_iter = 1000
    dqn = DQN(env=gym.make('CartPole-v0'), batch_size=64, gamma=0.99, eps_greedy=eps,
              model=dqn_model, replay_memory=ReplayMemory(100000),
              optimizer=dqn_optimizer, save_episodes=args.save, skip_k=args.k, max_iter=max_iter)

    dqn.run()

    fig, ax = plt.subplots()
    x = list(range(len(dqn.episode_durations)))
    ax.plot(x, dqn.episode_durations)
    ax.set_title("\n".join(wrap(str(eps) + str(dqn), 60)))
    plt.savefig(img_name)

    fig, ax = plt.subplots()
    ax.plot(x, dqn.action_values)
    ax.set_title("\n".join(wrap(str(eps) + str(dqn) + "OPT lr: " + str(lr), 60)))

    plt.savefig("avg_action_" + img_name)

