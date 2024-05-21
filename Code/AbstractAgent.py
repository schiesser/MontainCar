import gymnasium as gym
import numpy as np

class Agent:
    
    def __init__(self,env):
        self.starting_state, initial_observation = env.reset()
        self.action_space = env.action_space
        self.observations = []
        self.env = env
    
    def select_action(self, state):
        raise NotImplementedError
        
    def observe(self, state, action, next_state, reward):
        raise NotImplementedError
        
    def update(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError