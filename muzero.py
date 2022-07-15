import gym
import math
import random
import numpy as np
import torch.nn as nn
from copy import deepcopy

import torch
import torch.nn.functional as F
from network import MuZeroNetwork

device='cpu'

class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000, device=device):
    self.states  = np.zeros((length, obs_dim))
    self.actions = np.zeros((length, act_dim))
    self.rewards = np.zeros(length)
    self.nstates = np.zeros((length, obs_dim))
    self.values  = np.zeros(length)
    self.size = length
    self.idx  = 0

  def __len__(self): return self.idx

  def store(self,obs,policy,reward,n_obs,value):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = policy
    self.rewards[idx] = reward
    self.nstates[idx] = n_obs
    self.values[idx]  = value

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(device)
    rewards = torch.tensor( self.rewards[indices], dtype=torch.float).to(device)
    nstates = torch.tensor( self.nstates[indices], dtype=torch.float).to(device)
    values  = torch.tensor( self.values[indices], dtype=torch.float).to(device)
    return states, actions, rewards, nstates, values

class Node:
  def __init__(self, prior: float):
    self.prior = prior       # prior policy probabilities
    self.hidden_state = None # from dynamics function
    self.reward = 0          # from dynamics function
    self.policy = None       # from prediction function
    self.value_sum = 0       # from prediction function
    self.visit_count = 0
    self.children = {}
    #self.to_play = -1

  def value(self) -> bool:
    if self.visit_count == 0:
        return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> float:
    return len(self.children) > 0


class MuZero:
  def __init__(self, in_dims, out_dims):
    self.model  = MuZeroNetwork(in_dims, out_dims)
    self.memory = ReplayBuffer(in_dims, out_dims, device=device)

    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

  def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def ucb_score(self, parent: Node, child: Node, min_max_stats=None) -> float:
     """
     Calculate the modified UCB score of this Node. This value will be used when selecting Nodes during MCTS simulations.
     The UCB score balances between exploiting Nodes with known promising values, and exploring Nodes that haven't been 
     searched much throughout the MCTS simulations.
     """
     self.pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
     self.pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

     prior_score = self.pb_c * child.prior
     value_score = child.reward + self.discount * child.value()
     return prior_score + value_score

  def select_child(self, node: Node):
    total_visits_count = max(1 , sum([child.visit_count  for action, child in node.children.items()]) )
    action_index = np.argmax([self.ucb_score(node,child).detach().numpy() for action, child in node.children.items()])
    child  = node.children[action_index]
    action = np.array([1 if i==action_index else 0 for i in range(len(node.children))]) #.reshape(1,-1) 
    return action, child
    
  def mcts(self, obs, num_simulations=10, temperature=None):
    # init root node
    root = Node(0) 
    root.hidden_state = self.model.ht(obs)

    ## EXPAND root node
    policy, value = self.model.ft(root.hidden_state)
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])
      #root.children[i].to_play = -root.to_play

    # run mcts
    for _ in range(num_simulations):
      node = root 
      search_path = [node] # nodes in the tree that we select
      action_history = []  # the actions we took that got

      ## SELECT: traverse down the tree according to the ucb_score 
      while node.expanded():
        action, node = self.select_child(node)
        action_history.append(action)
        search_path.append(node)

      # EXPAND : now we are at a leaf which is not "expanded", run the dynamics model
      parent = search_path[-2]
      action = torch.tensor(action_history[-1])
 
      # run the dynamics model then use the ouput to predict a policy and a value
      node.reward, node.hidden_state = self.model.gt(torch.cat([parent.hidden_state,action],dim=0))
      policy, value = self.model.ft( node.hidden_state )

      # create all the children of the newly expanded node
      for i in range(policy.shape[0]):
        node.children[i] = Node(prior=policy[i])
        #node.children[i].to_play = -node.to_play

      # BACKPROPAGATE: update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value
        discount = 0.95
        value = bnode.reward + discount * value

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    total_num_visits = sum([ child.visit_count for action, child in root.children.items() ])
    policy = np.array( [ child.visit_count/total_num_visits for action, child in root.children.items()])

    if temperature == None: # take the greedy action (to be used during test time)
        action_index = np.argmax(policy)
    else: # otherwise sample (to be used during training)
        policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
        action_index = np.random.choice( np.arange(len(policy)) , p=policy )

    return policy, value, root

  def train(self, batch_size=128):
    lossfunc = nn.CrossEntropyLoss()
    if(len(self.memory) >= 256):
      obs, target_policys, target_rewards, n_obs, target_values = self.memory.sample(batch_size)
      lossFunc = torch.nn.BCELoss() # binary cross entropy 

      """
      lv,lp,lr = 0,0,0
      gradient_scale = 1
      num_unroll_steps = 5
      for i in range(0, num_unroll_steps+1):
        if i > 0:
          gradient_scale = 1/(num_unroll_steps)
          #values, rewards, policy_logits, states = network.recurrent_inference(states, actions[:,i-1].unsqueeze(-1))
          #hidden_states.register_hook(lambda grad: grad*0.5)
        states = self.model.ht(obs[i])
        policys, values  = self.model.ft(states)
        rewards, nstates = self.model.gt( torch.cat([states, policys],dim=0) )

        lv += lossFunc( F.softmax(values, dim=0), target_values[i]) * gradient_scale
        lr += lossFunc( F.softmax(rewards,dim=0), target_rewards[i])* gradient_scale
        lp += lossFunc( F.softmax(policys,dim=0), target_policys[i,:])* gradient_scale
      loss = lr+lv+lp
      """

      states = self.model.ht(obs)
      policys, values  = self.model.ft(states)
      rewards, nstates = self.model.gt( torch.cat([states, policys],dim=1) )

      lv = lossFunc( F.softmax(values, dim=0), target_values)
      lr = lossFunc( F.softmax(rewards,dim=0), target_rewards)
      lp = lossFunc( F.softmax(policys,dim=1), target_policys)
      loss = lr+lv+lp

      print(loss)
      #  https://github.com/werner-duvaud/muzero-general/blob/0825bd544fc172a2e2dcc96d43711123222c4a2f/trainer.py
      self.model.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
      self.model.optimizer.step()
    pass

def get_temperature(num_iter):
  # as num_iter increases, temperature decreases, and actions become greedier
  if num_iter < 100: return 3
  elif num_iter < 200: return 2
  elif num_iter < 300: return 1
  elif num_iter < 400: return .5
  elif num_iter < 500: return .25
  elif num_iter < 600: return .125
  else: return .0625

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = MuZero(env.observation_space.shape[0], env.action_space.n)

scores, time_step = [], 0
for epi in range(1000):
  obs = env.reset()
  while True:

    #env.render()
    policy, value, _ = agent.mcts(obs, 1, get_temperature(epi))
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)
    agent.memory.store(obs,policy,reward,n_obs,value)

    obs = n_obs
    agent.train()

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
