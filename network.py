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

class Network(nn.Module):
  def __init__(self, in_dims, hidden_size, out_dims, lr=1e-3):
    super().__init__()
    self.hid_dims = hidden_size
    self.out_dims = out_dims
    self.representation = nn.Sequential(
      nn.Linear(in_dims, 50), nn.ReLU(),
      nn.Linear(50, 50),      nn.ReLU(),
      nn.Linear(50, 50),      nn.ReLU(),
      nn.Linear(50, 50),      nn.ReLU(),
      nn.Linear(50, hidden_size)
    )

    self.dynamics = nn.Sequential(
      nn.Linear(hidden_size+out_dims, 50), nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, hidden_size+1) # add reward prediction
    )

    # actor critic network
    self.prediction = nn.Sequential(
      nn.Linear(hidden_size, 50), nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, 50),       nn.ReLU(),
      nn.Linear(50, out_dims+1) #  policy & value prediction
    )
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def ht(self, x):
    x = torch.tensor(x)
    return self.representation(x)


  def gt(self, x):
    out = self.dynamics(x)
    reward = out[:, -1]
    nstate = out[:, 0:self.hid_dims]
    return reward, nstate

  def ft(self, x):
    out = self.prediction (x)
    value  = out[:, -1]
    policy = out[:, 0:self.out_dims]
    policy = F.softmax(policy, dim=1)
    return policy, value

if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')

  in_dims  = env.observation_space.shape[0]
  out_dims = env.action_space.n
  hid_dims = 10
  mm = Network(in_dims,hid_dims,out_dims)

  for epi in range(10):
    score, _score = 0, 0
    obs,done = env.reset(), False
    while not done:
      state = mm.ht( [obs] )
      policy, value = mm.ft( state )
      action = policy.argmax().detach().numpy()
      print(state, [action])
      rew, nstate = mm.gt( torch.cat([state, policy],dim=1) )

      #print(value , rew)
      #action, rew = env.action_space.sample(), 0

      n_obs, reward, done, _ = env.step(action)
      score += reward
      _score += rew
      obs = n_obs
    print(f'Episode:{epi} Score:{score} Predicted score:{_score}')
  env.close()



