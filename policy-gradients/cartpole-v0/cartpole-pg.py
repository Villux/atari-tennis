import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--neurons', type=int, default=128, metavar='N',
                    help="Number of neurons in the hidden layer")
args = parser.parse_args()

print("Running the cartpole, with {} neurons and with discount rate of {}".format(args.neurons, args.gamma))

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
max_episodes = 5000

input_layer_size = 4
hidden_layer_size = args.neurons
output_layer_size = 2


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.affine2 = nn.Linear(hidden_layer_size, output_layer_size)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.multinomial()
    policy.saved_actions.append(action)
    return action.data


def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

episodes = []
average_timesteps = []
timesteps = []
running_reward = 10

for i_episode in range(0, max_episodes):
    state = env.reset()
    for t in range(10000):
        action = select_action(state)
        state, reward, done, _ = env.step(action[0,0])
        if args.render:
            env.render()
        policy.rewards.append(reward)
        if done:
            break
    timesteps.append(t)
    running_reward = running_reward * 0.99 + t * 0.01

    episodes.append(i_episode)
    average_timesteps.append(running_reward)

    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break


plt.plot(episodes, average_timesteps, label="Running average length", )
plt.scatter(episodes, timesteps, label="length", color="r")

plt.ylabel("Length")
plt.xlabel("Episodes")
plt.title("Discount: {}. Hidden layers: 1. Hidden neurons: {}.\n Total episodes: {}".format(args.gamma, hidden_layer_size, len(episodes)))
plt.legend()
plt.savefig("plots/cartpole_pg_1_layer_hidden_neurons_{}_gamma_{}_episodes_{}.png".format(hidden_layer_size, args.gamma, len(episodes)))




