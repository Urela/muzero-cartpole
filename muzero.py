import gym
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import *
from utils import *

device ='cpu'

class Node:
  def __init__(self, prior: float):
      self.prior = prior       # prior policy's probabilities

      self.reward = 0           # from dynamics function
      self.pred_policy = None   # from prediction function
      self.pred_value = None    # from prediction function
      self.hidden_state = None  # from dynamics function

      self.children = {}
      self.value_sum = 0    
      self.visit_count = 0

  # mean Q-value of this node
  def value(self) -> float:
    if self.visit_count == 0:
        return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> bool:
    return len(self.children) > 0


class muzero:
  def __init__(self, in_dims, out_dims):
    super(muzero, self).__init__()
    hid_dims = 10
    self.model = Network(in_dims,hid_dims,out_dims)

    self.memory = ReplayBuffer(100, out_dims)
    self.MinMaxStats = MinMaxStats()
    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    self.unroll_steps = 10     # number of timesteps to unroll to match action trajectories for each game sample
    self.bootstrap_steps = 500 # number of timesteps in the future to bootstrap true value
    pass

  def ucb_score(self, parent: Node, child: Node, min_max_stats=None) -> float:
    """
    Calculate the modified UCB score of this Node. This value will be used when selecting Nodes during MCTS simulations.
    The UCB score balances between exploiting Nodes with known promising values, and exploring Nodes that haven't been 
    searched much throughout the MCTS simulations.
    """
    self.pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    self.pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = self.pb_c * child.prior
    value_score = 0
    if child.visit_count > 0:
      if min_max_stats is not None:
        value_score = child.reward + self.discount * min_max_stats.normalize(child.value())
      else: value_score = child.reward + self.discount * child.value()
    return (prior_score + value_score).detach().numpy()

  def select_child(self, node: Node):
    total_visits_count = max(1 , sum([child.visit_count  for action, child in node.children.items()]) )
    action_index = np.argmax([self.ucb_score(node,child,self.MinMaxStats) for action, child in node.children.items()])
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

      # update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value #if root.to_play == bnode.to_play else -value
        self.MinMaxStats.update(node.value())
        value = bnode.reward + self.discount * value

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    total_num_visits = sum([ child.visit_count for action, child in root.children.items() ])
    policy = np.array( [ child.visit_count/total_num_visits for action, child in root.children.items()])

    if temperature == None: # take the greedy action (to be used during test time)
        action_index = np.argmax(policy)
    else: # otherwise sample (to be used during training)
        policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
        action_index = np.random.choice( np.arange(len(policy)) , p=policy )

    return policy, value, root

  def train(self, batch_size=10):
    mse = nn.MSELoss() # mean squard error
    bce = nn.BCELoss() # binary cross entropy 
    cre = nn.CrossEntropyLoss()
    if(len(self.memory) >= batch_size):
      data = self.memory.sample(10,5,0.5, batch_size)
     
      #OBS, ACTIONS, REWARDS, POLICYS, VALUES, 
      ## network unroll data
      #obs = torch.stack([np.flatten(data[i][0]) for i in range(batch_size)]).to(device).to(dtype) # flatten when insert into mem
      actions = np.stack([np.array(data[i][1], dtype=np.int64) for i in range(batch_size)])
      #
      ## targets
      rewards_target = torch.stack([torch.tensor(data[i][2]) for i in range(batch_size)]).to(device)#.to(dtype)
      policy_target  = torch.stack([np.stack( data[i][3]) for i in range(batch_size)]).to(device)#.to(dtype)
      value_target   = torch.stack([torch.tensor(data[i][4]) for i in range(batch_size)]).to(device)#.to(dtype) 

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0], env.action_space.n)


# self play
scores, time_step = [], 0
for epi in range(2000):
  obs = env.reset()

  game = Game(obs)
  agent.temperature = get_temperature(epi)

  while True:
    #env.render()
    policy, value, _ = agent.mcts(obs, 10)
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)

    game.store(obs,action,reward,done,policy,value)

    obs = n_obs
    agent.train(10)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      break
