import gym
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Network
from utils import *
        
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
    self.model = Network(in_dims, out_dims)
    self.memory = ReplayBuffer()
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

  def train(self, batch_size=100):
    mse = nn.MSELoss() # mean squard error
    bce = nn.BCELoss() # binary cross entropy 
    cre = nn.CrossEntropyLoss()
    if(len(self.memory) >= batch_size):

      # for every game in sample batch, unroll and update network weights 
      loss = 0
      for game in self.memory.sample(batch_size):

        game_length = len(game.rewards)
        index = np.random.choice(range(game_length)) # sampled index

        # We can only be unroll to the second-last time step, since every time step (index), 
        # we are predicting and matching values that are one time step into the future (index+1)
        if(index+self.unroll_steps < game_length): 
          unroll_steps = self.unroll_steps
        else: unroll_steps = game_length-1-index

        # 1) get hidden state from representation function
        # 2) iteratively:
        #  - feed hidden state into dynamics function and get a new hidden state with an asscoicated reward
        #  - feed this new hidden state into prediction function.
        state = self.model.ht(game.obss[index])
        for i in range(index, index+unroll_steps ):
          # inital state doesn't have a reward assoicated with it (as there is no previous action that led to it.)
          # So we run dynamics function to get a new state that has a reward.
          ### get predictions ###
          reward, state = self.model.gt(torch.cat([state, torch.tensor(game.policys[i]).float()],dim=0))
          policy, value = self.model.ft(state)

          ### make targets ###
          # bootstrap using transition rewards and mcts value for final bootstrapped time step
          if( game_length - i -1) >= self.bootstrap_steps:
            true_value = sum([ game.rewards[j] * (self.discount**(j-i)) for j in range(i, i+self.bootstrap_steps) ]) 
            true_value += game.values[i + self.bootstrap_steps] * ( self.discount**(self.bootstrap_steps) )
          # don't bootstrap; use only transition rewards until termination
          else: true_value = sum([ game.rewards[j] * ( self.discount**(j-i) ) for j in range(i, game_length) ])

          true_reward = game.rewards[i]   # since game.reward_history is shifted, this transition reward is actually at time step (start_index+1)
          true_policy = game.policys[i+1] # we need to match the pred_policy at time step (start_index+1) so we need to actually index game.policy_history at (start_index+1)

          ### calculate loss ###
          #loss += (1/unroll_steps) * ( mse(true_reward,reward) + mse(true_value,value) + bce(true_policy,policy) ) # take the average loss among all unroll steps
          loss += (1/unroll_steps) * ( (true_reward-reward)**2 + (true_value-value)**2 + cre(torch.tensor(true_policy),policy) ) # take the average loss among all unroll steps

      if loss >0:
        print(loss)
        self.model.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
        loss.backward()
        self.model.optimizer.step()
    pass



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
    policy, value, _ = agent.mcts(obs, 10, get_temperature(epi))
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)

    action = np.array([1 if i==action else 0 for i in range(len(policy))]).reshape(1,-1) 
    game.store(obs,reward,value,action,policy)

    obs = n_obs
    agent.train(10)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      break
