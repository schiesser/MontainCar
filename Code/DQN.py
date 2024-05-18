import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, info_per_state = 2, number_actions = 3):
        super(DQN, self).__init__()
        """
        input of neural : what characterizes a state
        output : one Q-value per possible action 
        !!! change later : consider number of hidden layers and number of neurons as hyperparameters
        """
        self.layer1 = nn.Linear(info_per_state, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, number_actions)
        
    def forward(self, x):
        """
        x (input of neural which as the size of "info_per_state") : can be one element or a batch
        defining activation functions
        !!! maybe later try with other activiation function ?
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class RND(nn.Module):

    def __init__(self, info_per_state = 2, number_actions = 3):
        super(RND, self).__init__()
        """
        input of neural : what characterizes a state
        output : one Q-value per possible action 
        !!! change later : consider number of hidden layers and number of neurons as hyperparameters
        """
        self.layer1 = nn.Linear(info_per_state, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        """
        x (input of neural which as the size of "info_per_state") : can be one element or a batch
        defining activation functions
        !!! maybe later try with other activiation function ?
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)