import random
from collections import deque

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size, dyna = False):
        random.shuffle(self.memory)
        if dyna:
            return list(self.memory)[:batch_size]
        list_batch = [list(self.memory)[int(i):int(i+batch_size)] for i in range(0, int(len(self.memory)), batch_size)]
        # attention when setting the batch size : avoid to have a small last batch
        return list_batch

    def __len__(self):
        return len(self.memory)