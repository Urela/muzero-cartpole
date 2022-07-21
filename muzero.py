import gym
import os
import numpy as np
from utils import *
from network import *

device = "cpu"

class Node:
  def __init__(self, prior: float):
      self.prior = prior       # prior policy's probabilities
      self.hidden_state = None  # from dynamics function
      self.reward = 0           # from dynamics function

      #self.policy = None # from prediction function
      #self.value = None # from prediction function
      #self.children = []
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
    #self.MinMaxStats = MinMaxStats()
    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    self.unroll_steps = 10     # number of timesteps to unroll to match action trajectories for each game sample
    self.bootstrap_steps = 500 # number of timesteps in the future to bootstrap true value
    pass

  def mcts(self, obs, num_simulations=100, temperature=1):

    # init root node
    root = Node(0) 
    root.hidden_state = self.model.ht(obs)

    ## EXPAND root node
    policy, value = self.model.ft(root.hidden_state)
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    # otherwise sample (to be used during training)
    total_num_visits = sum([ child.visit_count for action, child in root.children.items() ])
    policy = np.array( [ child.visit_count/total_num_visits for action, child in root.children.items()])


    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
    action = np.random.choice( np.arange(len(policy)) , p=policy ) # pick randomly using policy as distribution
    return action, policy, value, root


env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0], env.action_space.n)

# self play
scores, time_step = [], 0
for epi in range(1000):
  # since we set a seed for numpy, we can get reproducible results by 
  # setting a seed for the gym env, where the seed number is generated from numpy
  env.seed( int( np.random.choice( range(int(1e5)) ) ) ) 
  obs  = env.reset()
  game = Game(obs)

  while True:
    #env.render()
    action, policy, value, _ = agent.mcts(obs, 100, get_temperature(epi))

    #action = env.action_space.sample()
    n_obs, reward, done, info = env.step(action)

    game.store(obs,action,reward,policy,value)

    obs = n_obs
    agent.train(10)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      break
