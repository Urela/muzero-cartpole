
import gym
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Network
from utils import *
device = "cpu"
def get_temperature(num_iter):
  # as num_iter increases, temperature decreases, and actions become greedier
  if num_iter < 100: return 3
  elif num_iter < 200: return 2
  elif num_iter < 300: return 1
  elif num_iter < 400: return .5
  elif num_iter < 500: return .25
  elif num_iter < 600: return .125
  else: return .0625

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self):
    self.maximum = -float('inf')
    self.minimum = float('inf')

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class Game():
  def __init__(self, start_obs):
    self.obss  = [start_obs]
    self.rewards = [] # starts at t = 1 (the transition reward for reaching state 1)
    self.values  = [] # starts at t = 0 (from MCTS)
    self.policys = [] # starts at t = 0 (from MCTS)
    self.actions = [] # starts at t = 0
  def store(self,obs,reward,value,action,policy):
    self.obss.append(obs)
    self.rewards.append(reward)
    self.values.append(value)
    self.policys.append(policy)
    self.actions.append(action)

class ReplayBuffer():
  def __init__(self, size=1000):
    self.buffer = [] # list of Game objects, that contain the state, action, reward, MCTS policy, and MCTS value history
    self.buffer_size = size

  def __len__(self): return len(self.buffer)

  def store(self, game):
    if len(self.buffer) >= self.buffer_size: self.buffer.pop(0)
    self.buffer.append(game)
  def sample(self, batch_size=100): # sample a number of games from self.buffer, specified by the config parameter
    if len(self.buffer) <= batch_size: 
      return self.buffer.copy()
    return np.random.choice(self.buffer, size=batch_size,replace=False).tolist()

