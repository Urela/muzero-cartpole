import gym
import math
import numpy as np
import torch
import torch.nn.functional as F

from utils import *
from network import *

device = "cpu"

class Node:
  def __init__(self, prior: float):
      self.prior = prior       # prior policy's probabilities
      self.hidden_state = None  # from dynamics function
      self.reward = 0           # from dynamics function
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
    hid_dims =  256
    self.model = Network(in_dims, hid_dims, out_dims)
    self.memory = ReplayBuffer(size=1000)
    self.MinMaxStats = None
    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    self.root_dirichlet_alpha = 0.25
    self.root_exploration_fraction = 0.25

    self.unroll_steps = 10     # number of timesteps to unroll to match action trajectories for each game sample
    self.bootstrap_steps = 500 # number of timesteps in the future to bootstrap true value
    pass

  def ucb_score(self, parent: Node, child: Node) -> float:
    pb_c  = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = 0
    if child.visit_count > 0:
      value_score = child.reward + self.discount * self.MinMaxStats.normalize(child.value())
    return (prior_score + value_score).detach().numpy()

  def select_child(self, node: Node):
    total_visits_count = max(1 , sum([child.visit_count  for action, child in node.children.items()]) )
    action_index = np.argmax([self.ucb_score(node,child) for action, child in node.children.items()])
    child  = node.children[action_index]
    action = np.array([1 if i==action_index else 0 for i in range(len(node.children))]) #.reshape(1,-1) 
    return action, child


  def mcts(self, obs, num_simulations=100, temperature=1):
    # init root node
    root = Node(0) 
    root.hidden_state = self.model.ht([obs])

    ## EXPAND root node
    policy, value = self.model.ft(root.hidden_state)
    policy, value = policy.squeeze(), value.item()
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])

    # add exploration noise at the root
    actions = list(root.children.keys())
    noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(actions))
    frac = self.root_exploration_fraction
    for a, n in zip(actions, noise):
      root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac

    #  run mcts
    self.MinMaxStats = MinMaxStats() # re-initalize MinMaxStats
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
      action = torch.tensor(np.reshape(action_history[-1], (-1, 2)))

      node.reward, node.hidden_state = self.model.gt( torch.cat([parent.hidden_state,action],dim=1) )
      policy, value = self.model.ft( node.hidden_state )
      policy, value = policy.squeeze(), value.item()

      # create all the children of the newly expanded node
      for i in range(policy.shape[0]):
        node.children[i] = Node(prior=policy[i])

      # update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value #if root.to_play == bnode.to_play else -value
        self.MinMaxStats.update(node.value())
        value = bnode.reward + self.discount * value

    # output the final policy
    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    visit_counts = [x[1] for x in sorted(visit_counts)]
    av = torch.tensor(visit_counts, dtype=torch.float64)
    policy = F.softmax(av, dim=0).detach().numpy()
    return policy, value, root


env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0], env.action_space.n)

# self play
scores, time_step = [], 0
for epi in range(1000):
  # since we set a seed for numpy, we can get reproducible results by 
  # setting a seed for the gym env, where the seed number is generated from numpy
  #env.seed( int( np.random.choice( range(int(1e5)) ) ) ) 
  obs  = env.reset()
  game = Game(obs)

  while True:
    #env.render()
    policy, value, _ = agent.mcts(obs, 100, get_temperature(epi))
    action = policy.argmax()
    #action = env.action_space.sample()
    n_obs, reward, done, info = env.step(action)

    game.store(obs,action,reward,policy,value)

    obs = n_obs
    #agent.train(10)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      break
