import random
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

    def sample(self, batch_size, dyna = False):
        #randomize transition order
        random.shuffle(self.memory)
        
        if dyna:
            return list(self.memory)[:batch_size] #create a list of batch_size transitions shuffled

        #create multiple lists of batch length transitions. Transitions are not ordered
        list_batch = [list(self.memory)[int(i):int(i+batch_size)] for i in range(0, int(len(self.memory)), batch_size)]#last batch can be small
        
        return list_batch

    def __len__(self):
        #overrride len operation
        return len(self.memory)