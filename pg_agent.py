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
import torch.autograd as autograd
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

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 48)
        self.fc_output = nn.Linear(48, 2)
        self.sigmoid = nn.Sigmoid()

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.fc_output(hidden2)
        return self.sigmoid(output)

class PolicyGradient():
    def __init__(self, env, gamma, model, optimizer,
                 max_episodes=200, save_episodes=False, skip_k=0, max_iter=1000):
        self.env = env
        self.gamma = gamma
        self.model = model
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
        return 'POLICY GRAD: batch_size:' + ' gamma:' + str(self.gamma) + ' k:' + str(self.skip_k)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state))
        if random.random() < 0.2:
            action = LongTensor([[random.randrange(2)]])
            self.model.saved_actions.append(action)
            return action
        else:
            action = probs.data.max(1)[1].view(1,1)
            self.model.saved_actions.append(action)
            return action.data



    def optimize_model(self):
        R = 0
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for action, r in zip(self.model.saved_actions, rewards):
            action.reinforce(r)
        self.optimizer.zero_grad()
        autograd.backward(self.model.saved_actions, [None for _ in self.model.saved_actions])
        self.optimizer.step()

        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def handle_episode_results(self, duration):
        self.episode_durations.append(duration)
        if self.save_episodes:
            pickle.dump(self.episode_durations, open(file_name, 'wb'))
            pickle.dump(self.action_values, open(avg_action, 'wb'))

        if len(self.episode_durations) % 20 == 0:
            print(np.mean(self.episode_durations))


    def run(self):
        scores = deque(maxlen=100)
        for i in range(self.max_iter):
            state = self.env.reset()
            for t in count():
                action = self.select_action(state)
                import ipdb; ipdb.set_trace()
                next_state, reward, done, _ = self.env.step(action[0, 0])

                if done:
                    next_state = None
                    reward = -1

                self.model.rewards.append(reward)
                state = next_state
                self.steps_done += 1

                if done:
                    self.optimize_model()
                    score = t + 1
                    self.handle_episode_results(score)
                    scores.append(score)
                    break

            if np.mean(scores) >= 195 and i >= 100:
                return i


        self.env.close()



if __name__ == '__main__':
    args = parse_cl_args()

    policynet_model = PolicyNet()
    lr=1e-2
    dqn_optimizer = optim.Adam(policynet_model.parameters(), lr=lr)

    max_iter = 1000
    policy_gradient = PolicyGradient(env=gym.make('CartPole-v0'), gamma=0.99,
              model=policynet_model, optimizer=dqn_optimizer, save_episodes=args.save,
              skip_k=args.k, max_iter=max_iter)

    policy_gradient.run()

    fig, ax = plt.subplots()
    x = list(range(len(policy_gradient.episode_durations)))
    ax.plot(x, policy_gradient.episode_durations)
    ax.set_title("\n".join(wrap(str(policy_gradient), 60)))
    plt.savefig(img_name)

