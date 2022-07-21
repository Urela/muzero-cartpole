import numpy as np
#https://github.com/chiamp/muzero-cartpole/blob/master/classes.py

def get_temperature(num_iter):
  if num_iter < 100: return 3
  elif num_iter < 200: return 2
  elif num_iter < 300: return 1
  elif num_iter < 400: return .5
  elif num_iter < 500: return .25
  elif num_iter < 600: return .125
  else: return .0625

class Game: # wrapper for gym env
  def __init__(self,obs):

    self.current_state = obs
    self.states  = [obs.reshape(1,-1)] # starts at t = 0
    self.actions = [] # starts at t = 0
    self.rewards = [] # starts at t = 1 (the transition reward for reaching state 1)
    self.values  = [] # starts at t = 0 (from MCTS)
    self.policys = [] # starts at t = 0 (from MCTS)

  def store(self,obs, action, reward, policy, value):
    action = np.array([1 if i==action else 0 for i in range(len(policy))]).reshape(1,-1) 
    self.actions.append(action)
    self.rewards.append(reward)
    self.states.append(obs.reshape(1,-1))
    self.current_state = obs


class ReplayBuffer:
  def __init__(self,size=1000):
    # list of Game objects, that contain the state, action, reward, MCTS policy, and MCTS value history
    self.buffer = [] 
    self.buffer_size = size

  def store(self,game):
    if len(self.buffer) >= self.buffer_size: self.buffer.pop(0)
    self.buffer.append(game)

  # sample a number of games from self.buffer, specified by the config parameter
  def sample(self,batch_size): 
    if len(self.buffer) <= batch_size: return self.buffer.copy()
    return np.random.choice(self.buffer,size=batch_size,replace=False).tolist()

if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')

  mem = ReplayBuffer(size = 1000)
  obs, done = env.reset(), False
  game, score = Game(obs), 0
  while not done:
    policy,value = np.array((1, env.action_space.n)), 0
    action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)
    game.store(obs,action,reward,policy,value)
    score +=reward

    obs = n_obs
  print(score)
  mem.store(game)
  env.close()

