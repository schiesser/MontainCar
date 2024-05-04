import random
from collections import namedtuple, deque

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        random.shuffle(self.memory)
        list_batch = [self.memory[i:i+batch_size] for i in range(0, len(self.memory), batch_size)]
        # attention when setting the batch size : avoid to have a small last batch
        return list_batch

    def __len__(self):
        return len(self.memory)