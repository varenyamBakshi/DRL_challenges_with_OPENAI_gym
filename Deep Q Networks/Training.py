import Wrappers
from model import DQN
import argparse
import time
import numpy as np
import collections
import re

import torch
from torch import nn, optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
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
        return np.array(states), np.array(actions, dtype=np.int64), np.array(rewards,dtype=np.float32), np.array(is_dones, dtype=np.uint8), np.array(next_states) 

        
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

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, is_dones, next_states = batch

    states_v = torch.Tensor(states).to(device)
    next_states_v = torch.Tensor(next_states).to(device)
    actions_v = torch.Tensor(actions).to(device=device, dtype=torch.int64)
    rewards_v = torch.Tensor(rewards).to(device)
    done_mask = torch.Tensor(is_dones).to(device, dtype=torch.bool) 
    state_action_values = net(states_v).gather(1,actions_v.unsqueeze(-1)).squeeze(-1)
    
   
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
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=None, help="file name to extract parameters from")
    parser.add_argument("--cuda",default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the Environment, default = "
    + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND, 
    help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = Wrappers.make_env(args.env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    write_mode = "w" # by default open the file in write mode
    initial_games = 0
    if args.params:
        print("loading parameters from file-"+str(args.params))
        net.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
        net.load_state_dict(net.state_dict())
        write_mode = "a"
        initial_games = int(args.params.split('-')[1]) 

    writer = SummaryWriter("plots-"+args.env+"/run2")
    outFile = open("Output_Records2.txt", write_mode) # to store the terminal output records to file
    print(net)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env,buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0   # counter of frames
    ts_frame = 0    # counter of the frame in which we get some reward
    ts = time.time() # to keep track of time
    best_mean_reward = None

    while True: 
        frame_idx += 1
        if initial_games==0:
            epsilon = max(EPSILON_FINAL, EPSILON_START-frame_idx/EPSILON_DECAY_LAST_FRAME)
        else: epsilon = EPSILON_FINAL
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None: # it returns a non-None value only when the episode ends
            total_rewards.append(reward)
            games_played = initial_games + len(total_rewards)
            speed = (frame_idx-ts_frame)/(time.time()-ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            out_text = "%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s"%(frame_idx,games_played,mean_reward, epsilon,speed)
            print(out_text)
            outFile.write(out_text+"\n")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), "agent/run2/"+args.env+"-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved"%(best_mean_reward,mean_reward)) 
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("solved in %d frames!"%frame_idx)
                break
            if games_played==1 or games_played%50==0:
                torch.save(net.state_dict(), "agent/run2/history/After-"+str(games_played)+"-games.dat")
        if len(buffer) < REPLAY_START_SIZE: continue 
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict()) # syncing the target network with current training network
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
    outFile.close()



