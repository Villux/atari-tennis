import argparse
import numpy as np 
from itertools import count

# OpenaAi-Gym
import gym
# PyTorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import os.path

resume = True

parser = argparse.ArgumentParser(description='PyTorch implementation of OpenAi-Gym Atari 2600 Tennis-ram-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log_interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()

args.cuda = not args.disable_cuda and torch.cuda.is_available()
args.cuda = False

env = gym.make('Tennis-ram-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

input_layer_size = 128
output_layer_size = 18
hidden_layer1_size = 50
hidden_layer2_size = 50


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # Affine refers to affine transformation, which is a Linear transformation (f(x) = Ax + b) where the term b is 0.
        # In neural networks the b denotes bias.

        # In other words self.affine1 and self.affine2 are the weight matrices between the layers.
        self.affine1 = nn.Linear(input_layer_size, hidden_layer1_size)
        self.affine2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)        
        self.affine3 = nn.Linear(hidden_layer2_size, output_layer_size)

        # Policy gradient needs to store actions and rewards to calculate the backpropagation
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        # The input layer is multiplied by the first weight matrix self.affine1. 
        # I.e the weight matrix between the input layer and the hidden layer.
        # The activation function is a Rectifier. ReLu = Rectified Linear unit
        # 
        # One way ReLUs improve neural networks is by speeding up training. The gradient 
        # computation is very simple (either 0 or 1 depending on the sign of x)
        # Logistic and hyperbolic tangent activation functions suffer from vanishing gradient problem,  
        # which makes learning increasingly slower. 
        
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x)) 

        # The hidden layer is multiplied by the second weight matrix self.affine2.
        # The activation function is softmax function, which is a sigmoid function, but 
        # the terms of the vector are normalized so that the sum of the terms of the vector action_scores = 1.
        action_scores = F.softmax(self.affine3(x))
        return action_scores

policy = Policy()
episodes = []
running_rewards = []
running_reward = None

if resume == True and os.path.exists("save_tennis_ram_model.p"):
    policy.load_state_dict(torch.load("save_tennis_ram_model.p"))
    if os.path.exists("save_tennis_ram_bookkeeping.p"):
        temp = torch.load("save_tennis_ram_bookkeeping.p")
        episodes = temp['episodes']
        running_rewards = temp['running_rewards']
        running_rewad = running_rewards[-1]
    
    
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

if args.cuda:
    policy.cuda()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    if args.cuda: state = state.cuda()
    probs = policy(Variable(state))

    # Draw action from the probability distribution of the output. 
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
    if args.cuda: rewards = rewards.cuda()
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]


state = env.reset()



reward_sum = 0
episode_number = len(episodes)
frame = 0

while True:
    if args.render: env.render()

    action = select_action(state)
    state, reward, done, _ = env.step(action[0,0])
    reward_sum += reward
    policy.rewards.append(reward)
    frame += 1


    if done: 
        episode_number += 1
        finish_episode()

        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = 0.99 * running_reward + 0.01 * reward_sum

        print("Episode: {}. Resetting the environment. Episode reward total was {}. Running mean: {}".format(episode_number, reward_sum, running_reward))

        episodes.append(episode_number)
        running_rewards.append (running_reward)

        if episode_number != 0 and episode_number % 100 == 0:
            torch.save(policy.state_dict(), "save_tennis_ram_model.p")
            torch.save({'running_rewards': running_rewards, 'episodes': episodes}, 'save_tennis_ram_bookkeeping.p')

        reward_sum = 0
        frame = 0
        state = env.reset()



    #if reward != 0:
        #print("Ep. {}. Frame: {}. Round finished!, reward: {}".format(episode_number, frame, reward))

