import gym
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cpu"

def get_temperature(episodes):
  # as episodes increases, temperature decreases, and actions become greedier
  if episodes < 250: return 1
  elif episodes < 300: return 0.75
  elif episodes < 400: return 0.65
  elif episodes < 500: return 0.55
  elif episodes < 600: return 0.3
  else: return 0.25

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

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
    self.actions = [] # starts at t = 0
    self.rewards = [] # starts at t = 1 (the transition reward for reaching state 1)
    self.dones   = []
    self.policys = [] # starts at t = 0 (from MCTS)
    self.values  = [] # starts at t = 0 (from MCTS)
  def store(self,obs,action,reward,done,policy,value):
    self.obss.append(obs)
    self.actions.append(action)
    self.rewards.append(reward)
    self.dones.append(done)
    self.policys.append(policy)
    self.values.append(value)
  def __len__(self): return len(self.rewards)

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

