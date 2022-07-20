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
  def __init__(self, size, num_actions):
    self.size = size
    self.buffer = [] # list of Game objects, that contain the state, action, reward, MCTS policy, and MCTS value history
    self.num_actions = num_actions

  def __len__(self): return len(self.buffer)

  def store(self, game):
    if len(self.buffer) >= self.size: self.buffer.pop(0)
    self.buffer.append(game)

  def _sample(self, unroll_steps, n, discount):
    #  n = n-step-return 
    # select trajectory
    index = np.random.choice(range(len(self.buffer))) # sampled index
    game_length = len(self.buffer[index]) # get the length of a game
    last_index  = game_length - 1

    # select start index to unroll
    start_index = np.random.choice(game_length)

    ## fill in the data
    OBS = self.buffer[index].obss[start_index]
    VALUES, REWARDS, POLICYS, ACTIONS = [],[],[],[]

    for step in range(start_index, start_index+unroll_steps+1 ):
      n_index = step + n
      if n_index >= game_length:
        value = torch.tensor([0]).float().to(device)
      else: value = self.buffer[index].values[n_index] * (discount ** n) # discount value

      # add discounted rewards until step n or end of episode
      last_valid_index = np.minimum(last_index, n_index)
      for i, reward in enumerate(self.buffer[index].rewards[step:last_valid_index]):
      #for i, reward in enumerate(self.memory[memory_index]["rewards"][step::]): # rewards until end of episode
        value += reward * (discount ** i)
      VALUES.append(value)

      # only add when not inital step | dont need reward for step 0
      if step != start_index:
        if step > 0 and step <= last_index:
          REWARDS.append( self.buffer[index].rewards[step-1] ) 
        else: REWARDS.append( 0 ) 

      # add policy
      if step > 0 and step <= last_index:
        POLICYS.append( self.buffer[index].policys[step] ) 
      else: 
        #for mse loss
        POLICYS.append( np.repeat(1,self.num_actions)/self.num_actions ) 

        #for cross entropy loss
        #POLICYS.append( torch.tensor(np.repeat(1,self.num_actions)/0)) 

    # unroll steps beyond trajectory then fill in the remaining (random) actions
    last_valid_index = np.minimum(last_index - 1, start_index + unroll_steps - 1)
    num_steps = last_valid_index - start_index

    # real
    ACTIONS = self.buffer[index].actions[start_index:start_index+num_steps+1]
   
    # fills
    for _ in range(unroll_steps - num_steps + 1):
      ACTIONS.append(np.random.choice(np.arange(self.num_actions)))
    return OBS, ACTIONS, REWARDS, POLICYS, VALUES, 

  def sample(self, unroll_steps, n, discount, batch_size=100):
    #OBS, ACTIONS, REWARDS, POLICYS, VALUES, 
    data = [ (self._sample(unroll_steps,n,discount)) for _ in range(batch_size)]
    return data

