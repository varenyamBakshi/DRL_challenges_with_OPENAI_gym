import Wrappers
import model
import argparse
import time
import numpy as np
import collections

import torch
from torch import nn, optim

from tensorboardX import SummaryWriter

env_name = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5
GAMMA = 0.99
BATCH_SIZE = 32     #batch size sampled from replay buffer for one iteration
REPLAY_SIZE = 10000     # max capacity of buffer
REPLAY_START_SIZE = 10000 # count of frames to wait for before training
                          # to populate the replay buffer  
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000 # how frequently we sync model weights from training
                          # model to the target model which is used to get the value 
                          # for next state in Bellman approximation

EPSILON_DECAY_LAST_FRAME = 10**5 # epsilon decays till this frame
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = collections.namedtuple('Experience',
 field_names=['state','action','reward','is_done','new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,replace=False)
        states, actions, rewards, is_dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards,dtype=np.float32), np.array(is_dones), np.array(next_states) 

        
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self,net, epsilon=0.0, device='cpu'):
        done_reward=None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()

        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.Tensor(state_a).to(device)
            q_vals_v = net(state_v) # pass through the network to obtain q-values
            _, act_v = torch.max(q_vals_v, dim=1) # select the action with 
            action = int(act_v.item())            # with highest q-value
    
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(self, batch, net, tgt_net, device="cpu"):
        states, actions, rewards, is_dones, next_states = batch

        states_v = torch.Tensor(states).to(device)
        next_states_v = torch.Tensor(next_states).to(device)
        actions_v = torch.Tensor(actions).to(device)
        rewards_v = torch.Tensor(rewards).to(device)
        done_mask = torch.ByteTensor(is_dones).to(device)
        
        state_action_values = net(states_v).gather(1,
        actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = tgt_net(next_states_v).max(1)[0] # .max() returns both max and argmax
        next_state_values[done_mask] = 0.0 # if it is the last step of episode so 
                                           # discounted reward from next state is zero
                                           # without this the training would not converge
        next_state_values = next_state_values.detach()
        # we detach the the value from computation graph to prevent gradients from 
        # flowing into neural network to calculate Q approximations for next states
        # it returns the tensor without the connection to its calculation history

        expected_state_action_values = next_state_values*GAMMA +rewards_v
        return nn.MSELoss()(state_action_values,expected_state_action_values)
        
        

