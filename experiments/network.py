import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# https://github.com/Hauf3n/MuZero-PyTorch/blob/master/Networks.py

# representation:  s_0 = h(o_1, ..., o_t)
# dynamics:        r_k, s_k = g(s_km1, a_k)
# prediction:      p_k, v_k = f(s_k)

"""
Network contains the representation, dynamics and prediction network.
These networks are trained during agent self-play.
"""
device = 'cpu'

class Network(nn.Module):
  def __init__(self, in_dims, hidden_size, out_dims, lr=1e-3):
    super().__init__()
    self.hid_dims = hidden_size
    self.out_dims = out_dims
    self.representation = nn.Sequential(
      nn.Linear(in_dims, 256), nn.ReLU(),
      nn.Linear(256, 256),     nn.ReLU(),
      nn.Linear(256, hidden_size)
    ).to(device)

    self.dynamics = nn.Sequential(
      nn.Linear(hidden_size+1, 256), nn.ReLU(),
      nn.Linear(256, 256),           nn.ReLU(),
      nn.Linear(256, hidden_size+1) # add reward prediction
    ).to(device)

    # actor critic network
    self.prediction = nn.Sequential(
      nn.Linear(hidden_size, 256), nn.ReLU(),
      nn.Linear(256, 256),         nn.ReLU(),
      nn.Linear(256, out_dims+1) #  policy & value prediction
    ).to(device)

    #self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.optimizer = optim.Adam(
      list(self.representation.parameters()) +\
      list(self.dynamics.parameters()) +\
      list(self.prediction.parameters()),
      lr=lr
    )

  def ht(self, x):
    return self.representation(x)


  def gt(self, x):
    out = self.dynamics(x)
    reward = out[:, -1]
    nstate = out[:, 0:self.hid_dims]
    return nstate, reward

  def ft(self, x):
    out = self.prediction (x)
    value  = out[:, -1]
    policy = out[:, 0:self.out_dims]
    policy = F.softmax(policy, dim=1)
    return policy, value

if __name__ == '__main__':
  import gym
  from utils import *
  env = gym.make('CartPole-v1')
  env = gym.wrappers.RecordEpisodeStatistics(env)

  history_length = 3
  stack_obs = stack_observations(history_length)

  in_dims  = env.observation_space.shape[0]*history_length
  out_dims = env.action_space.n
  hid_dims = 50
  mm = Network(in_dims,hid_dims,out_dims)


  scores = []
  for epi in range(10):
    obs = env.reset()
    obs = stack_obs(obs)
    while True:
      state = mm.ht( torch.tensor(obs, dtype=torch.float) )
      policy, value = mm.ft( state )
      action = policy.argmax().detach()
      bb = state.shape[0]
      _action = torch.tensor(action,dtype=torch.float).reshape(bb,1) / out_dims
      rew, nstate = mm.gt( torch.cat([state, _action],dim=1) )

      obs, reward, done, info = env.step(action.numpy())
      obs = stack_obs(obs)

      if "episode" in info.keys(): 
        scores.append(int(info['episode']['r']))
        avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
        print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
        break

  env.close()



