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
    self.prior = prior        # prior policy's probabilities
    self.hidden_state = None  # from dynamics function
    self.reward = 0           # from dynamics function

    self.children = {}
    self.value_sum = 0    
    self.visit_count = 0

  def value(self) -> float: # mean Q-value of this node
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> bool:
    return len(self.children) > 0


class muzero:
  def __init__(self, in_dims, out_dims):
    super(muzero, self).__init__()
    hid_dims = 50
    self.model = Network(in_dims,hid_dims,out_dims)

    self.memory = ReplayBuffer(100, out_dims)
    self.MinMaxStats = MinMaxStats()
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

  def mcts(self, obs, num_simulations=10, temperature=None):
    # init root node
    root = Node(0) 
    root.hidden_state = self.model.ht([obs])

    ## EXPAND root node
    policy, value = self.model.ft(root.hidden_state)
    for i in range(policy.shape[1]):
      root.children[i] = Node(prior=policy[:,i])

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
      # one hot encoding 
      action = torch.zeros_like(policy)
      action[:,action_history[-1]] = 1

      node.reward, node.hidden_state = self.model.gt( torch.cat([parent.hidden_state,action],dim=1) )
      policy, value = self.model.ft( node.hidden_state )

      # create all the children of the newly expanded node
      for i in range(policy.shape[1]):
        node.children[i] = Node(prior=policy[:,i])

      # update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value #if root.to_play == bnode.to_play else -value
        self.MinMaxStats.update(node.value())
        value = bnode.reward + self.discount * value

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    #total_num_visits = sum([ child.visit_count for action, child in root.children.items() ])
    #policy = np.array([ child.visit_count/total_num_visits for action, child in root.children.items() ])

    #policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
    #action_index = np.random.choice( np.arange(len(policy)) , p=policy )

    # output the final policy
    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    visit_counts = [x[1] for x in sorted(visit_counts)]
    av = torch.tensor(visit_counts, dtype=torch.float64)
    policy = F.softmax(av, dim=0).detach().numpy()

    return policy, value, root

  def train(self, batch_size=10):
    mse = nn.MSELoss() # mean squard error
    bce = nn.BCELoss() # binary cross entropy 
    if(len(self.memory) >= batch_size):
            
            
      unroll_steps = 5
      reward_coef = 1
      value_coef = 1
      n = 10
      for i in range(16):
        self.model.optimizer.zero_grad()
        data = self.memory.sample(unroll_steps, n, self.discount, batch_size)

        ## network unroll data
        obs = torch.stack([torch.flatten(torch.tensor(data[i][0])) for i in range(batch_size)]).to(device).float()#.to(dtype) # flatten when insert into mem
        actions = np.stack([np.array(data[i][1], dtype=np.int64) for i in range(batch_size)])
         
        ## targets
        rewards_target = torch.stack([torch.tensor(data[i][2]) for i in range(batch_size)]).to(device).float()#.to(dtype)
        policy_target  = torch.stack([torch.tensor(np.stack( data[i][3])) for i in range(batch_size)]).to(device).float()#.to(dtype)
        value_target   = torch.stack([torch.tensor(data[i][4]) for i in range(batch_size)]).to(device).float()#.to(dtype) 

        # loss
        loss = torch.tensor(0).float().to(device)
        
        # agent inital step
        states = self.model.ht(obs)
        policys, values = self.model.ft(states)
         
        #policy_loss = torch.mean(torch.sum(- policy_target[:,0].detach() * logsoftmax(p), 1)) # policy cross entropy
        policy_loss = mse(policys, policy_target[:,0].detach())   # policy mse
        value_loss  = mse(values, value_target[:,0].detach())
        
        loss += (policy_loss + value_coef * value_loss) / 2
        # steps
        for step in range(1, unroll_steps+1):
          step_action = actions[:,step - 1]
          #state, p, v, rewards = agent.rollout_step(state, step_action)

          rewards, states = self.model.gt(torch.cat([states, policys],dim=1))
          policys, values = self.model.ft(states)

          #policy_loss = torch.mean(torch.sum(- policy_target[:,0].detach() * logsoftmax(p), 1)) # policy cross entropy
          policy_loss = mse(policys, policy_target[:,step].detach())   # policy mse
          value_loss  = mse(values, value_target[:,step].detach())
          reward_loss = mse(rewards, rewards_target[:,step-1].detach())
          loss += ( policy_loss + value_coef * value_loss + reward_coef * reward_loss) / unroll_steps

        #print(loss)
        loss.backward()
        self.model.optimizer.step() 

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0], env.action_space.n)


# self play
scores, time_step = [], 0
for epi in range(2000):
  obs = env.reset()
  game = Game(obs)

  while True:
    #env.render()
    policy, value, _ = agent.mcts(obs, 20, get_temperature(epi))
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)
    game.store(obs,action,reward,done,policy,value)

    obs = n_obs
    agent.train(32)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      break
