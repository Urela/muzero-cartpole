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

    self.unroll_steps = 10     # number of timesteps to unroll to match action trajectories for each game sample
    self.bootstrap_steps = 500 # number of timesteps in the future to bootstrap true value
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

      reward_coef = 1
      value_coef = 1
      n = 10
      unroll_steps = 5
      discount = 0.99
      dtype = torch.float
      for _ in range(1):
        self.model.optimizer.zero_grad()
        # Load a batch of random games
        sample_obs, sample_actions, sample_rewards, sample_policys, sample_values =  self.memory.sample3(batch_size) # returns a batch of games 

        game_lengths = np.array([ len(g) for g in sample_obs ])
        game_last_indices = game_lengths - 1

        # select start index to unroll and fill data
        start_indices = rng.integers(low=0, high=game_lengths)

        obs = torch.tensor(([ sample_obs[i][idx] for i, idx in enumerate(start_indices)] )).to(device).to(dtype)

        # unroll steps beyond trajectory then fill in the remaining (random) actions
        last_valid_indices = np.minimum(game_last_indices - 1, start_indices + unroll_steps - 1)
        num_steps = last_valid_indices - start_indices
        num_fills = unroll_steps - num_steps + 1 

        actions = [sample_actions[i][idx:idx+step+1] for i,(idx, step) in enumerate(zip(start_indices, num_steps))] 
        for i, n in enumerate(num_fills):
          for _ in range(n):
            actions[i].append(np.random.choice(self.out_dims))  # fill in fake actions
        actions = np.vstack(actions)

        # compute n-step return for every unroll step, rewards and pi
        rewards_target, value_target, policy_target = [],[],[] 
        for i, start_index in enumerate(start_indices):
          rewards, values, policys = [], [], []
          for step in range(start_index, start_index+unroll_steps+1 ):

            #########
            n_index = step + n
            if n_index >= game_last_indices[i]:
              value = torch.tensor([0], dtype=torch.float).to(device)
            else: value = sample_values[i][n_index] * (self.discount ** n) # discount value

            # add discounted rewards until step n or end of episode
            last_valid_index = np.minimum(game_last_indices[i], n_index)
            for j, reward in enumerate(sample_values[i][step:last_valid_index]):
              value += reward * (self.discount ** j)
            values.append(value)
            #########

            # only add when not inital step | dont need reward for step 0
            if step != start_index:
              if step > 0  and step <= game_last_indices[i]:
                rewards.append(sample_rewards[i][step-1])
              else:
                rewards.append(torch.tensor([0.0]).to(device))
            # add policy
            if step >= 0  and step < game_last_indices[i]:
              policys.append(sample_policys[i][step])
            else: policys.append(torch.tensor(np.ones(self.out_dims)/self.out_dims)) # mse loss

          rewards_target.append(rewards)
          policy_target.append(torch.stack(policys))
          value_target.append(values)

        rewards_target = torch.tensor( rewards_target ).to(device).to(dtype) 
        value_target = torch.tensor( value_target )    .to(device).to(dtype)  
        policy_target = torch.stack( policy_target )  .to(device).to(dtype) 
        ### loss
        loss = torch.tensor(0).to(device).to(dtype)

        ## agent inital step
        states = self.model.ht(obs)
        policys, values = self.model.ft(states)
        ##  
        ##policy mse
        policy_loss = mse(policys, policy_target[:,0].detach())
        value_loss  = mse(values, value_target[:,0].detach())
        loss += ( policy_loss + value_coef * value_loss) / 2

        ### steps
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

history_length = 3
stack_obs = stack_observations(history_length)

env = gym.make('CartPole-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = muzero(env.observation_space.shape[0]*history_length, env.action_space.n)

# self play
scores, time_step = [], 0
for epi in range(3):
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
      agent.train(3)
      break
