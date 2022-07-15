import torch
import torch.nn as nn
import torch.optim as optim

# representation:  s_0 = h(o_1, ..., o_t)
# dynamics:        r_k, s_k = g(s_km1, a_k)
# prediction:      p_k, v_k = f(s_k)

class MuZeroNetwork(nn.Module):
  def __init__(self, in_dims, out_dims):
    super().__init__()
    hidden_state = in_dims


    self.representation_net  = nn.Sequential(
      nn.Linear(in_dims, 64), nn.ReLU(),
      nn.Linear(64, 64),      nn.ReLU(),
      nn.Linear(64, hidden_state)
    )

    self.dynamics_net  = nn.Sequential(
      nn.Linear(hidden_state+out_dims, 64), nn.ReLU(),
      nn.Linear(64, 64),      nn.ReLU(),
    )
    self.dyn_reward = nn.Linear(64, 1)
    self.dyn_nstate = nn.Linear(64, hidden_state)
    
    self.prediction_net  = nn.Sequential(
      nn.Linear(in_dims, 64), nn.ReLU(),
      nn.Linear(64, 64),      nn.ReLU(),
    )
    self.pred_value  = nn.Linear(64, 1)
    self.pred_policy = nn.Linear(64, out_dims)

    self.optimizer = optim.Adam(self.parameters(), lr=0.001)

  def ht(self, obs):  
    obs = torch.tensor(obs)
    x = self.representation_net(obs)
    return x

  def gt(self, state_action):        
    x = self.dynamics_net(state_action)
    reward = self.dyn_reward(x).squeeze().detach()
    nstate = self.dyn_nstate(x)
    return reward, nstate

  def ft(self, state):    
    x = self.prediction_net(state)
    value  = self.pred_value(x).squeeze().detach()
    policy = self.pred_policy(x)
    return policy, value

if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')

  mm = MuZeroNetwork(env.observation_space.shape[0], env.action_space.n)

  for epi in range(10):
    score, _score = 0, 0
    obs,done = env.reset(), False
    while not done:
      state = mm.ht( obs )
      policy, value = mm.ft( state )
      action = policy.argmax().detach().numpy()
      rew, nstate = mm.gt(torch.cat([state, policy],dim=0) )
      print(value )

      #action, rew = env.action_space.sample(), 0

      n_obs, reward, done, _ = env.step(action)
      score += reward
      _score += rew
      obs = n_obs
    print(f'Episode:{epi} Score:{score} Predicted score:{_score}')
  env.close()



