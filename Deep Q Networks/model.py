import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape,n_actions):
        super(DQN,self).__init__()
        self.input_shape = input_shape[0]
        self.conv1 = nn.Conv2d(self.input_shape,32,kernel_size=8, stride=4),
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        conv_out_shape = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_shape,512)
        self.fc2 = nn.Linear(512,n_actions)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def _get_conv_out(self,shape): # runs convolution operators on dummy tensor to get the output shape
        test = torch.zeros(1, *shape)
        o = self.conv2(self.conv1(test))
        return int(np.prod(o.size()))