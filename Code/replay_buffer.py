import random
import numpy as np
from collections import deque

class ReplayMemory(object):
#class to store transitions.

    def __init__(self, capacity, Transition):
        #list with max capacity
        self.memory = deque([], maxlen=capacity)
        # Transition = ( state, next state, action, reward)
        self.Transition = Transition

    def push(self, *args):
        #add a transition to memory
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        #randomize transition order
        random.shuffle(self.memory)

        #create multiple lists of batch length transitions. Transitions are not ordered
        list_batch = [list(self.memory)[int(i):int(i+batch_size)] for i in range(0, int(len(self.memory)), batch_size)]#last batch can be small
        
        return list_batch

    def __len__(self):
        #overrride len operation
        return len(self.memory)
    




class ReplayMemoryDyna(object):

    def __init__(self, capacity, Transition):
        self.memory = list()
        self.capacity = capacity
        self.Transition = Transition

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
        else:
            position = np.random.randint(0, self.capacity)
            self.memory[position] = self.Transition(*args)

    def sample(self):
        return self.memory

    def __len__(self):
        return len(self.memory)