import random
import gym
from collections import namedtuple
import numpy as np
import torch
from torch import nn, optim
import tensorboard
from tensorboardX import SummaryWriter


hidden_size = 128   # size of hidden input layer
batch_size = 100  # number of episodes played in one iteration
percentile = 30   # we shall take only top 30 percentile episodes for training
gamma = 0.9 #discount factor

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self,env):
        super(DiscreteOneHotWrapper,self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype = np.float32)

    def observation(self,observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


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
    #assign rewards accourding to the number of steps taken to complete
    disc_rewards = list(map(lambda s: s.reward*(gamma**len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)
    mean_disc_rewards = np.mean(disc_rewards)

    train_obs = []
    train_act = []
    elite_batch=[] #as the successful cases are very rare, we need to store them for a longer time
    for episode,reward in zip(batch,disc_rewards):
        if reward > reward_bound:
            #separating the observations and actions for training purpose
            train_obs.extend(map(lambda step: step.observation, episode.steps))
            train_act.extend(map(lambda step: step.action, episode.steps))
            elite_batch.append(episode)
        
    #train_obs_v = torch.tensor(train_obs, dtype=torch.float32)
    #train_act_v = torch.tensor(train_act, dtype=torch.int64)
    return elite_batch, train_obs, train_act, reward_bound, mean_disc_rewards

if __name__ == "__main__":
    random.seed(101)
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    #env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n #--------

    network = Net(obs_size, hidden_size, n_actions)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=0.005) 
    lr_optimizer = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.80)
    writer = SummaryWriter('frozenLake/exp5', comment="lr decay debuggedp")

    full_batch = []
    for iter_num, batch in enumerate(iterate_batches(env, network, batch_size)):
        percent_success = float(np.mean(list(map(lambda s: s.reward,batch))))

        full_batch, obs, acts, reward_b, mean_reward = filter_batch(full_batch+batch, percentile)
        if not full_batch: continue
        obs_v = torch.tensor(obs, dtype = torch.float32)
        acts_v = torch.tensor(acts, dtype = torch.int64)
        full_batch = full_batch[-500:]#keeping only last 500 elite episodes
        #training the neural network
        optimizer.zero_grad()
        action_scores_v = network(obs_v)
        loss_v = loss_func(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        lr_optimizer.step()
        lr = lr_optimizer.get_lr()[0]

        print("%d: loss=%.3f, percent_success=%.1f, reward_bound=%.1f, mean_rewards=%.3f, learning rate=%.4f" 
        % (iter_num, loss_v.item(), percent_success, reward_b, mean_reward, lr))
        #for plotting on tensorboardX
        writer.add_scalar("loss", loss_v.item(), iter_num)
        writer.add_scalar("success rate", percent_success, iter_num)
        writer.add_scalar("reward_bound", reward_b, iter_num)
        writer.add_scalar("mean reward", mean_reward,iter_num)
        if percent_success > 0.8:
            print("solved!")
            break
    writer.close()