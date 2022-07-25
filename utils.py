
import random
import torch
import numpy as np
from collections import deque
device ="cpu"

def get_temperature(episodes):
  # as episodes increases, temperature decreases, and actions become greedier
  if episodes < 250: return 1
  elif episodes < 300: return 0.75
  elif episodes < 400: return 0.65
  elif episodes < 500: return 0.55
  elif episodes < 600: return 0.3
  else: return 0.25

class MinMaxStats():
  def __init__(self):
    self.max = - np.inf
    self.min = np.inf
      
  def update(self, value):
    self.max = np.maximum(self.max, value.cpu())
    self.min = np.minimum(self.min, value.cpu())
      
  def normalize(self, value):
    value = value.cpu()
    if self.max > self.min:
      return ((value - self.min) / (self.max - self.min)).to(device)
    return value

class stack_observations:
  def __init__(self, history_length):
    self.obs_stack = deque(maxlen=history_length)
    self.history_length = history_length
  def __call__(self,obs):
    self.obs_stack.append(obs)
    features = np.zeros((self.history_length, len(obs)))
    
    # features 
    size = len(self.obs_stack)
    if size == self.history_length:
      features = np.array(self.obs_stack)
    else:
      features[self.history_length-size::] = np.array(self.obs_stack)
    return features.flatten().reshape(1,-1)

class Game():
  def __init__(self, start_obs):
    self.obs  = [torch.tensor(start_obs)]
    self.actions = [] # starts at t = 0
    self.rewards = [] # starts at t = 1 (the transition reward for reaching state 1)
    self.policys = [] # starts at t = 0 (from MCTS)
    self.values  = [] # starts at t = 0 (from MCTS)
  def store(self,obs,action,reward,policy,value):
    #self.obs.append(torch.tensor(obs))
    self.rewards.append(torch.tensor(reward))
    self.policys.append(torch.tensor(policy))
    self.obs.append(obs)
    self.actions.append(action)
    #self.rewards.append(reward)
    #self.policys.append(policy)
    self.values.append(value)
  def __len__(self): return len(self.obs)

class ReplayBuffer():
  def __init__(self, size, num_actions):
    # list of Game objects, that contain the state, action, reward, MCTS policy, and MCTS value history
    self.buffer = deque(maxlen=size)
    self.size = size
    self.num_actions = num_actions

    self.obs     = deque(maxlen=size) 
    self.actions = deque(maxlen=size) 
    self.rewards = deque(maxlen=size)
    self.policys = deque(maxlen=size)
    self.values  = deque(maxlen=size) 

  def __len__(self): 
    return len(self.buffer)

  def store(self, game):
    self.buffer.append(game)

  def store3(self, game):
    self.obs.append( game.obs)
    self.actions.append(game.action)
    self.rewards.append(game.rewards)
    self.policys.append(game.policys)
    self.values.append(game.value)

  def _sample(self, unroll_steps, n, discount):
    sample = {}
    sample["obs"], sample["pi"], sample["v"], sample["actions"], sample["rewards"], sample["return"] = [],[],[],[],[],[]

    # select trajectory
    mem_idx = np.random.choice(range(len(self.buffer))) # sampled index

    game_length = len(self.buffer[mem_idx]) # get the length of a game
    game_last_index = game_length - 1

    # select start index to unroll and fill data
    start_index = np.random.choice(game_length)
    print(game_length)

    sample["obs"] = self.buffer[mem_idx].obs[start_index]

    # compute n-step return for every unroll step, rewards and pi
    for step in range(start_index, start_index+unroll_steps+1 ):
      n_index = step + n

      if n_index >= game_last_index:
        value = torch.tensor([0], dtype=torch.float).to(device)
      else: value = self.buffer[mem_idx].values[n_index] * (discount ** n) # discount value

      # add discounted rewards until step n or end of episode
      last_valid_index = np.minimum(game_last_index, n_index)
      for i, reward in enumerate(self.buffer[mem_idx].rewards[step:last_valid_index]):
        value += reward * (discount ** i)
      sample["return"].append(value)

      # add reward
      # only add when not inital step | dont need reward for step 0
      if step != start_index:
        if step > 0  and step <= game_last_index:
          sample["rewards"].append(self.buffer[mem_idx].rewards[step-1])
        else:
          sample["rewards"].append(torch.tensor([0.0]).to(device))
          
      # add policy
      if step >= 0  and step < game_last_index:
        sample["pi"].append(self.buffer[mem_idx].policys[step])
      else:
        sample["pi"].append(torch.tensor(np.repeat(1,self.num_actions)/self.num_actions)) # mse loss
        #sample["pi"].append(torch.tensor(np.repeat(1,self.num_actions)*0.0)) # cross entropy loss

    # unroll steps beyond trajectory then fill in the remaining (random) actions
    last_valid_index = np.minimum(game_last_index - 1, start_index + unroll_steps - 1)
    num_steps = last_valid_index - start_index
    
   
    num_fills = unroll_steps - num_steps + 1 

    sample["actions"] = self.buffer[mem_idx].actions[start_index:start_index+num_steps+1]    # real
    for i in range(num_fills):
      sample["actions"].append(np.random.choice(self.num_actions))  # fills
    
    return sample

  def sample(self, unroll_steps, n, discount, batch_size=100):
    return [ (self._sample(unroll_steps,n,discount)) for _ in range(batch_size)]

  def sample2(self, batch_size=100): # sample a number of games from self.buffer, specified by the config parameter
    if len(self.buffer) <= batch_size: 
      return self.buffer.copy()
    return np.random.choice(self.buffer, size=batch_size,replace=False).tolist()

  def sample3(self, batch_size=100): # sample a number of games from self.buffer, specified by the config parameter
    game_batchs = random.sample(self.buffer, batch_size)
    #obs     = [ np.vstack(g.obs)     for g in game_batchs ]
    obs     = [ np.vstack(g.obs)     for g in game_batchs ]
    actions = [ g.actions for g in game_batchs ] 
    rewards = [ g.rewards for g in game_batchs ] 
    policys = [ g.policys for g in game_batchs ] 
    values  = [ g.values  for g in game_batchs ] 
    return obs, actions, rewards, policys , values
    #return self.obs[batch], self.actions[batch], self.rewards[batch], self.policys[batch], self.values[batch]

if __name__ == '__main__':
  import gym

  env = gym.make('CartPole-v1')
  env = gym.wrappers.RecordEpisodeStatistics(env)

  history_length = 3
  stack_obs = stack_observations(history_length)

  in_dims  = env.observation_space.shape[0]
  out_dims = env.action_space.n
  memory = ReplayBuffer(100, env.action_space.n)

  scores = []
  for epi in range(10):
    obs = env.reset() 
    game = Game(obs)
    while True:

      action = env.action_space.sample()
      policy, value = np.array([1 if i==action else 0 for i in range(out_dims)]), 0

      obs, reward, done, info = env.step(action)
      game.store(obs,action,reward,done,policy,value)

      if "episode" in info.keys(): 
        scores.append(int(info['episode']['r']))
        avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
        print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")

        memory.store(game)
        break
    memory.sample(5, 10, 0.99, 100)
  env.close()
