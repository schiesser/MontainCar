import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, info_per_state = 2, number_actions = 3):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Linear(info_per_state, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, number_actions)
        
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class RND(nn.Module):

    def __init__(self, info_per_state = 2, nb_neurons = 16):
        super(RND, self).__init__()

        self.layer1 = nn.Linear(info_per_state, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)