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

class Representation(nn.Module):
  def __init__(self, in_dims, hidden_size):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(in_dims, 50), nn.ReLU(),
      nn.Linear(50, 50),      nn.ReLU(),
      nn.Linear(50, 50),      nn.ReLU(),
      nn.Linear(50, 50),      nn.ReLU(),
      nn.Linear(50, hidden_size)
    )

  def forward(self, x):
    x = torch.tensor(x)
    return self.network(x)

class Dynamics(nn.Module):
  def __init__(self, hidden_size, out_dims):
    super().__init__()
    self.hid_dims = hidden_size
    self.network = nn.Sequential(
      nn.Linear(hidden_size+out_dims, 50), nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, hidden_size+1) # add reward prediction
    )

  def forward(self, x):
    out = self.network(x)
    nstate = out[0:self.hid_dims]
    reward = out[-1]
    return reward, nstate

class Prediction(nn.Module):
  def __init__(self, hidden_size, out_dims):
    super().__init__()
    self.out_dims = out_dims
    self.network = nn.Sequential(
      nn.Linear(hidden_size, 50), nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, out_dims+1) # # value & policy prediction
    )

  def forward(self, x):
    out = self.network(x)
    value  = out[-1]
    policy = out[0:self.out_dims]
    policy = F.softmax(policy, dim=0)
    return policy, value

if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')

  in_dims  = env.observation_space.shape[0]
  out_dims = env.action_space.n
  hid_dims = 10

  ht = Representation(in_dims, hid_dims)
  ft = Prediction(hid_dims, out_dims)
  gt = Dynamics(hid_dims, out_dims)
  

  for epi in range(10):
    score, _score = 0, 0
    obs,done = env.reset(), False
    while not done:
      state = ht( obs )
      policy, value = ft( state )
      rew, nstate = gt( torch.cat([state, policy],dim=0) )

      action = policy.argmax().detach().numpy()
      #print(value , rew)
      #action, rew = env.action_space.sample(), 0

      n_obs, reward, done, _ = env.step(action)
      score += reward
      _score += rew
      obs = n_obs
    print(f'Episode:{epi} Score:{score} Predicted score:{_score}')
  env.close()



