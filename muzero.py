import gym
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from network import *
from utils import *
rng = np.random.default_rng()

device = "cpu"

class Node:
  def __init__(self, prior: float):
    self.prior = prior       # prior policy's probabilities
    self.hidden_state = None 
    self.reward = 0           

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
    hid_dims = 50

    self.model = Network(in_dims,hid_dims,out_dims)

    self.memory = ReplayBuffer(1000, out_dims)
    self.MinMaxStats = None 
    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    self.root_dirichlet_alpha = 0.25
    self.root_exploration_fraction = 0.25

    self.out_dims = out_dims
    pass

  def rollout_step(self, state, action): 
    # unroll a step
    batch_size = state.shape[0]
    action = torch.tensor(action,dtype=torch.float).to(device).reshape(batch_size,1) / self.out_dims
    nstate, reward = self.model.gt( torch.cat([state,action],dim=1) )

    policy, value = self.model.ft(nstate)
    return nstate, reward, policy, value

  def ucb_score(self, parent: Node, child: Node) -> float:
    pb_c  = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = 0
    if child.visit_count > 0:
      value_score = self.MinMaxStats.normalize( child.reward + self.discount * child.value())
    return prior_score + value_score


  def select_child(self, node: Node):
    action = np.argmax([self.ucb_score(node,child) for action, child in node.children.items()])
    child  = node.children[action]
    return action, child

  def mcts(self, obs, num_simulations=10, temperature=1):
    # init root node
    root = Node(0) 
    root.hidden_state = self.model.ht(torch.tensor(obs,dtype=torch.float))

    ## EXPAND root node
    policy, _ = self.model.ft(root.hidden_state)  # random inital policy
    policy = policy.detach().cpu()
    for i in range(policy.shape[1]):
      root.children[i] = Node(prior=policy[0,i])

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
      action = action_history[-1]
      nstate, reward, policy, value = self.rollout_step(parent.hidden_state, [action])
      nstate, reward, policy, value = nstate.detach(), reward.detach(), policy.detach(), value.detach()
      
      node.hidden_state = nstate
      node.reward = reward

      # create all the children of the newly expanded node
      for i in range(policy.shape[1]):
        node.children[i] = Node(prior=policy[0,i])

      # update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value
        self.MinMaxStats.update( bnode.reward + self.discount * bnode.value())
        value = bnode.reward + self.discount * value

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    total_num_visits = sum([ child.visit_count for _, child in root.children.items() ])

    policy = np.array([ child.visit_count/total_num_visits for _, child in root.children.items() ])
    policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
    action = np.random.choice(len(policy), p=policy)
    policy = torch.tensor(policy).float()

    value = root.value()
    return action, policy, value, root

  def train(self, batch_size=32):
    mse = nn.MSELoss()          # mean squard error
    cre = nn.CrossEntropyLoss() # cross entropy loss
    if(len(self.memory) >= batch_size):

      reward_coef, value_coef = 1, 1
      unroll_steps, n = 5, 10
      discount = 0.99
      for _ in range(16):
        self.model.optimizer.zero_grad()
        data = self.memory.sample(unroll_steps, n, discount, batch_size)
        # network unroll data
        obs = torch.stack(data["obs"]).to(device).to(torch.float) # flatten when insert into mem
        actions = np.stack(data["actions"])

        # targets
        rewards_target = torch.stack( data["rewards"]).to(device).to(torch.float)
        policy_target  = torch.stack( data["policys"]).to(device).to(torch.float)
        value_target   = torch.stack( data["returns"]).to(device).to(torch.float)
        
        # loss
        loss = torch.tensor(0).to(device).to(torch.float)

        # agent inital step
        states = self.model.ht(obs)
        policys, values = self.model.ft(states)

        #policy mse
        policy_loss = mse(policys, policy_target[:,0].detach())
        value_loss  = mse(values, value_target[:,0].detach())
        loss += ( policy_loss + value_coef * value_loss) / 2

        # steps
        for step in range(1, unroll_steps+1):
          step_action = actions[:,step - 1]
          states, rewards, policys, values = self.rollout_step(states, step_action)

          #policy mse
          policy_loss = mse(policys, policy_target[:,step].detach())
          value_loss  = mse(values, value_target[:,step].detach())
          reward_loss = mse(rewards, rewards_target[:,step-1].detach())
          
          loss += ( policy_loss + value_coef * value_loss + reward_coef * reward_loss) / unroll_steps
        ##print(loss)
        loss.backward()
        self.model.optimizer.step() 
    pass

history_length = 1
stack_obs = stack_observations(history_length)

env = gym.make('CartPole-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0]*history_length, env.action_space.n)

# self play
scores, time_step = [], 0
for epi in range(1000):
  obs = env.reset()
  obs = stack_obs(obs)
  game = Game(obs)

  while True:
    #env.render()
    action, policy, value, root = agent.mcts(obs, 20, get_temperature(epi))
    obs, reward, done, info = env.step(action)
    obs = stack_obs(obs)
    game.store(obs,action,reward,policy,value)

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      agent.memory.store(game)
      agent.train(32)
      break
