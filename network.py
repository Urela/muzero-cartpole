import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# representation:  s_0 = h(o_1, ..., o_t)
# dynamics:        r_k, s_k = g(s_km1, a_k)
# prediction:      p_k, v_k = f(s_k)

"""
Network contains the representation, dynamics and prediction network.
These networks are trained during agent self-play.
"""

class Network(nn.Module):
  def __init__(self, in_dims, out_dims):
    super().__init__()
    hidden_state = 256 # hidden state space size

    # Representation net  
    self.rep1 = nn.Linear(in_dims, 256)
    self.rep2 = nn.Linear(256, hidden_state)

    # Dynamics net
    self.dyn_input  = nn.Linear(hidden_state+out_dims, 256)
    self.dyn_reward = nn.Linear(256, 1)
    self.dyn_nstate = nn.Linear(256, hidden_state)

    # Prediction net
    self.pred_input  = nn.Linear(hidden_state, 256)
    self.pred_value  = nn.Linear(256, 1)
    self.pred_policy = nn.Linear(256, out_dims)

    self.optimizer = optim.Adam(self.parameters(), lr=0.001)

  def ht(self, obs):  
    x = torch.tensor(obs)
    x = F.relu(self.rep1(x))
    x = F.relu(self.rep2(x))
    return x

  def gt(self, state_action):        
    x = self.dyn_input(state_action)
    reward = F.relu(self.dyn_reward(x))
    nstate = F.relu(self.dyn_nstate(x))
    return reward.item(), nstate

  def ft(self, state):    
    x = self.pred_input(state)
    value  = F.relu(self.pred_value(x))
    policy = F.relu(self.pred_policy(x))
    return policy, value.item()

if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')

  mm = Network(env.observation_space.shape[0], env.action_space.n)

  for epi in range(10):
    score, _score = 0, 0
    obs,done = env.reset(), False
    while not done:
      state = mm.ht( obs )
      policy, value = mm.ft( state )
      rew, nstate = mm.gt( torch.cat([state, policy],dim=0) )

      action = policy.argmax().detach().numpy()
      #print(value , rew)
      #action, rew = env.action_space.sample(), 0

      n_obs, reward, done, _ = env.step(action)
      score += reward
      _score += rew
      obs = n_obs
    print(f'Episode:{epi} Score:{score} Predicted score:{_score}')
  env.close()



