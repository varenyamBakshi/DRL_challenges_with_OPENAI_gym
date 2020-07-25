import Wrappers
import model
import argparse
import time
import numpy as np

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

