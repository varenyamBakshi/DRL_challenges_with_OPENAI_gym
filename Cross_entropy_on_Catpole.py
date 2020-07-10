#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

hidden_size = 128   # size of hidden input layer
batch_size = 16   # number of episodes played in one iteration
percentile = 70   # we shall take only top 30 percentile episodes for training

class Net(nn.Module):
    def __init__(self,obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_actions)
        )
            # created a simple neural network with one hidden layer

    def forward(self,x):
        return self.net(x)

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size):
    batch =[]      # to store batches of episodes
    episode_reward =0.0
    episode_steps = []  # episode steps for current episode
    sm = nn.Softmax(dim=1)
    obs = env.reset()   # reset the environment to get first observation

    while True:
        obs_v = torch.tensor([obs],dtype=torch.float32)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]  #since its a two dimensional array we extract the first element only
        #choosing a random action using the policy
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, __ = env.step(action)
        
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation = obs, action = action))   # note: add the observation which we used to
                                                                                #choose the action and not the one returned after the action                 
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch #once desired no. of episodes are accumulated it return the batch to train the network
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for episode in batch:
        if episode.reward < reward_bound: continue
        #separating the observations and actions for training purpose
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))
    
    train_obs_v = torch.tensor(train_obs, dtype=torch.float32)
    train_act_v = torch.tensor(train_act, dtype=torch.int64)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n #--------

    network = Net(obs_size, hidden_size, n_actions)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_num, batch in enumerate(iterate_batches(env, network, batch_size)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, percentile)
        #training the neural network
        optimizer.zero_grad()
        action_scores_v = network(obs_v)
        loss_v = loss_func(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_num, loss_v.item(), reward_m, reward_b))
        #for plotting on tensorboardX
        writer.add_scalar("loss", loss_v.item(), iter_num)
        writer.add_scalar("reward_bound", reward_b, iter_num)
        writer.add_scalar("reward_mean", reward_m, iter_num)

        if reward_m > 199:
            print("solved!")
            break
    writer.close()