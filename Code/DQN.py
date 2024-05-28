import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, info_per_state = 2, number_actions = 3, nb_neurons=64):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Linear(info_per_state, nb_neurons)
        self.layer2 = nn.Linear(nb_neurons, nb_neurons)
        self.layer3 = nn.Linear(nb_neurons, number_actions)
        
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class RND(nn.Module):

    def __init__(self, info_per_state = 2, nb_neurons=32):
        super(RND, self).__init__()

        self.layer1 = nn.Linear(info_per_state, nb_neurons)
        self.layer2 = nn.Linear(nb_neurons, nb_neurons)
        self.layer3 = nn.Linear(nb_neurons, 1)
        
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)