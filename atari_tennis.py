import gym
from PIL import Image
import numpy as np

def rgb2gray(image):
    return image[:,:,0] * 299/1000 + image[:,:,1] * 587/1000 + image[:,:,2] * 114/1000

def prepro(I):
    import ipdb; ipdb.set_trace()
    # Keep only needed part of the image and make it a squere
    I = I[48: 212, 10: 150]
    # Downsample to (75, 75)
    I = I[::2, ::2]
    # Grayscale and return
    return rgb2gray(I).ravel()

def policy_forward(input):
    return

def policy_backwards():
    return




env = gym.make('Tennis-v0')
observation = env.reset()

for i in range(10000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    prepro(observation)


img = Image.fromarray(I, 'RGB')
img.show()
env.close()



# TODO
# Policy forward
# Stockhastic policy - which action to choose?

# Policy backward