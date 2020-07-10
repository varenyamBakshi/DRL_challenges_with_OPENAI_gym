import gym
from collections import namedtuple
import numpy as np
import torch
from torch import nn, optim
from tensorboardx import SummaryWriter

hidden_size = 128
batch_size = 16
percentile = 70   # we shall take only top 30 percentile episodes for training

