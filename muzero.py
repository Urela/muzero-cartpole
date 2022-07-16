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

    ## output the final policy
    #visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    #visit_counts = [x[1] for x in sorted(visit_counts)]
    #av = torch.tensor(visit_counts, dtype=torch.float64)
    #policy = F.softmax(av, dim=0).detach().numpy()
    return policy, value, root

  def train(self, batch_size=100):
    crossL = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    if(len(self.memory) >= 1):
      #obs, target_policys, target_rewards, n_obs, target_values = self.memory.sample(batch_size)
      lossFunc = torch.nn.BCELoss() # binary cross entropy 
      for game in self.memory.sample(1):
        lv,lp,lr = 0,0,0
        gradient_scale = 1
        num_unroll_steps = 5
        for i in range(0, num_unroll_steps+1):
          state = self.model.ht(game.obss[i])
          policy, value  = self.model.ft(state)
          reward, nstate = self.model.gt( torch.cat([state, policy],dim=0) )

          #print( policy, torch.tensor(game.policys[i]))
          lr += (reward - game.rewards[i]) **2
          lv += (value - game.values[i]) **2
          lp += crossL( policy, torch.tensor(game.policys[i]))* gradient_scale


          #print(value, game.values[i]) 
          #print(reward, game.rewards[i]) 
          #lr += mse(reward, game.rewards[i]) * gradient_scale
          #lv += mse(value,  game.values[i]) * gradient_scale
          #lp += crossL( policy, game.policys[i])* gradient_scale

      loss = lr+lv+lp
      print(loss)

      """
        # first we get the hidden state representation using the representation function
        # then we iteratively feed the hidden state into the dynamics function with the corresponding action, as well as feed the hidden state into the prediction function
        # we then match these predicted values to the true values
        # note we don't call the prediction function on the initial hidden state representation given by the representation function, since there's no associating predicted transition reward to match the true transition reward
        # this is because we don't / shouldn't have access to the previous action that lead to the initial state

      # First we get hidden state using the representation function
      # then we iteratilvely feed the hidden states into the dynamics function with the corresponding policies as well as feed the hidden states into the prediction function

      loss = lr+lv+lp
      print(loss)
      """
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
  game = Game(obs)
  while True:
    #env.render()
    policy, value, _ = agent.mcts(obs, 100, get_temperature(epi))
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)

    game.obss.append(obs)
    game.rewards.append(reward)
    game.actions.append( np.array([1 if i==action else 0 for i in range(len(policy))]).reshape(1,-1) )
    game.values.append(value)
    game.policys.append(policy)

    obs = n_obs
    agent.train()

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
  agent.memory.store(game)
