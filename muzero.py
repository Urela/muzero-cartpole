import gym
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Network
        
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

class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=1000, device=device):
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

  def sample(self, batch_size=100):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(device)
    rewards = torch.tensor( self.rewards[indices], dtype=torch.float).to(device)
    nstates = torch.tensor( self.nstates[indices], dtype=torch.float).to(device)
    values  = torch.tensor( self.values[indices], dtype=torch.float).to(device)
    return states, actions, rewards, nstates, values

"""
A class that represents the nodes used in Monte Carlo Tree Search.
"""
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
    self.memory = ReplayBuffer(in_dims, out_dims)
    self.MinMaxStats = MinMaxStats()
    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25
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
        bnode.value_sum += value #if root.to_play == bnode.to_play else -value
        bnode.visit_count += 1
        #min_max_stats.update(node.value())
        value = bnode.reward + self.discount * value

    # output the final policy
    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    visit_counts = [x[1] for x in sorted(visit_counts)]
    av = torch.tensor(visit_counts, dtype=torch.float64)
    policy = F.softmax(av, dim=0).detach().numpy()
    return policy, value, root

  def train(self, batch_size=128):
    lossfunc = nn.CrossEntropyLoss()
    if(len(self.memory) >= 256):
      obs, target_policys, target_rewards, n_obs, target_values = self.memory.sample(batch_size)
      lossFunc = torch.nn.BCELoss() # binary cross entropy 

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
      """

      print(loss)
      #  https://github.com/werner-duvaud/muzero-general/blob/0825bd544fc172a2e2dcc96d43711123222c4a2f/trainer.py
      self.model.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
      self.model.optimizer.step()
    pass



env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0], env.action_space.n)


# self play
scores, time_step = [], 0
for epi in range(1000):
  obs = env.reset()
  while True:

    #env.render()
    policy, value, _ = agent.mcts(obs, 100, get_temperature(epi))
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
