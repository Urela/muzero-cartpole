import gym
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import *
from utils import *
from itertools import chain
        
device = "cpu"

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
    self.ht = Representation(in_dims, hid_dims)
    self.ft = Prediction(hid_dims, out_dims)
    self.gt = Dynamics(hid_dims, out_dims)

    self.memory = ReplayBuffer()
    self.MinMaxStats = MinMaxStats()
    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25
    self.unroll_steps = 10     # number of timesteps to unroll to match action trajectories for each game sample
    self.bootstrap_steps = 500 # number of timesteps in the future to bootstrap true value
    self.temperature = 1
    pass

  @property
  def parameters(self):
    return chain(self.ht.parameters(), self.gt.parameters(), self.ft.parameters())

  def ucb_score(self, parent: Node, child: Node) -> float:
    pb_c  = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = 0
    if child.visit_count > 0:
      value_score = child.reward + self.discount * self.MinMaxStats.normalize(child.value())
    return (prior_score + value_score).item()

  def select_child(self, node: Node):
    total_visits_count = max(1 , sum([child.visit_count  for action, child in node.children.items()]) )
    action_index = np.argmax([self.ucb_score(node,child) for action, child in node.children.items()])
    child  = node.children[action_index]
    action = np.array([1 if i==action_index else 0 for i in range(len(node.children))]) #.reshape(1,-1) 
    return action, child

  def mcts(self, obs, num_simulations=10):
    # init root node
    root = Node(0) 
    root.hidden_state = self.ht(obs)

    ## EXPAND root node
    policy, value = self.ft(root.hidden_state)
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])

    root_dirichlet_alpha = 0.25
    root_exploration_fraction = 0.25

    # add exploration noise at the root
    actions = list(root.children.keys())
    noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
    frac = root_exploration_fraction
    for a, n in zip(actions, noise):
      root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac

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
      node.reward, node.hidden_state = self.gt(torch.cat([parent.hidden_state,action],dim=0))
      policy, value = self.ft( node.hidden_state )

      # create all the children of the newly expanded node
      for i in range(policy.shape[0]):
        node.children[i] = Node(prior=policy[i])

      # update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value 
        self.MinMaxStats.update(bnode.reward + self.discount*node.value())
        value = bnode.reward + self.discount * value

    # output the final policy
    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    visit_counts = [x[1] for x in sorted(visit_counts)]
    av = np.array(visit_counts).astype(np.float64)
    policy = softmax(av)
    return policy, value, root

  def train(self, batch_size=100):
    mse = nn.MSELoss() 
    cel = nn.CrossEntropyLoss()
    lsm = nn.LogSoftmax()

    if(len(self.memory) >= batch_size):
      #if episode < 250:   agent.temperature = 1
      #elif episode < 300: agent.temperature = 0.75
      #elif episode < 400: agent.temperature = 0.65
      #elif episode < 500: agent.temperature = 0.55
      #elif episode < 600: agent.temperature = 0.3
      #else: agent.temperature = 0.25
      optimizer = optim.Adam(self.parameters(), lr=lr)

      # for every game in sample batch, unroll and update network weights 
      loss = 0
      for game in self.memory.sample(batch_size):

        game_length = len(game.rewards)
        index = np.random.choice(range(game_length)) # sampled index



env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0], env.action_space.n)


# self play
scores, time_step = [], 0
for epi in range(1000):
  obs = env.reset()
  game = Game(obs)
  while True:
    #env.render()
    policy, value, _ = agent.mcts(obs, 10)
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)

    action = np.array([1 if i==action else 0 for i in range(len(policy))])
    game.store(obs,reward,value,action,policy)

    obs = n_obs
    agent.train(10)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      break
