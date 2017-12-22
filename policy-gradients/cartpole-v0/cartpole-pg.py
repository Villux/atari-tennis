#!/usr/bin/env python

import numpy as np
import gym
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed = 123

parser = argparse.ArgumentParser(description='OpenAi Gym cartpole with policy gradients')

parser.add_argument('--gamma', type=float, default=0.99, metavar='g', help='discount factor for rewards')
parser.add_argument('--render', action='store_true', help='renders the game')
parser.add_argument('--neurons', type=int, default=100, metavar='N', help="Number of neurons in the hidden layer")
args = parser.parse_args()

print("Running the cartpole, with {} neurons and with discount rate of {}".format(args.neurons, args.gamma))

torch.manual_seed(seed)

max_episodes = 5000

input_layer_size = 4
hidden_layer_size = args.neurons
output_layer_size = 2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Single layered network:
        # W1 is the weights between input and hidden layer
        # W2 is the weights between hidden layer and output
        self.W1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.W2 = nn.Linear(hidden_layer_size, output_layer_size)

        self.actions = []
        self.rewards = []

    def forward(self, x):
        return F.softmax(self.W2(F.relu(self.W1(x))))


policy_nn = NeuralNetwork()
optimizer = optim.Adam(policy_nn.parameters(), lr=1e-2)


def update_weights():
    reward = 0
    rewards = []
    for r in policy_nn.rewards[::-1]:
        reward = r + args.gamma * reward
        rewards.insert(0, reward)
    rewards = torch.Tensor(rewards)
    
    # Standardize the rewards
    rewards -= rewards.mean()
    rewards /= rewards.std() + np.finfo(np.float32).eps

    for action, r in zip(policy_nn.actions, rewards):
        action.reinforce(r)

    optimizer.zero_grad()
    autograd.backward(policy_nn.actions, [None for _ in policy_nn.actions])
    optimizer.step()
    
    del policy_nn.rewards[:]
    del policy_nn.actions[:]



def sample_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # Get the probability distribution of actions
    action_probabilities = policy_nn(Variable(state))
    # Sample that distribution
    action = action_probabilities.multinomial()
    policy_nn.actions.append(action)
    return action.data

env = gym.make('CartPole-v0')
env.seed(seed)

episodes_list = []
running_average_episode_length_list = []
episode_length_list = []
running_average_episode_length = 10

for episode in range(1, max_episodes):
    
    state = env.reset()

    for episode_length in range(1000):
        action = sample_action(state)
        state, reward, done, _ = env.step(action[0,0])
        
        if args.render:
            env.render()

        policy_nn.rewards.append(reward)

        if done:
            break

    episode_length_list.append(episode_length)
    running_average_episode_length = running_average_episode_length * 0.99 + episode_length * 0.01

    episodes_list.append(episode)
    running_average_episode_length_list.append(running_average_episode_length)

    update_weights()

    if episode % 10 == 0:
        print('Episode number: {} Length: {} Running average length: {}'.format(episode, episode_length, running_average_episode_length))
    if running_average_episode_length > env.spec.reward_threshold:
        print("Game solved! Reached the reward threshold, with running average length: {}. Final episode length: {}".format(running_average_episode_length, episode_length))
        break


plt.plot(episodes_list, running_average_episode_length_list, label="Running average length", )
plt.scatter(episodes_list, episode_length_list, label="length", color="r")

plt.ylabel("Length")
plt.xlabel("Episodes")
plt.title("Discount: {}. Hidden layers: 1. Hidden neurons: {}.\n Total episodes: {}".format(args.gamma, hidden_layer_size, len(episodes_list)))
plt.legend()
plt.savefig("plots/cartpole_pg_1_layer_hidden_neurons_{}_gamma_{}_episodes_{}.png".format(hidden_layer_size, args.gamma, len(episodes_list)))




